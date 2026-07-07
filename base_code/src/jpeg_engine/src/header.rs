/// JPEG marker types
#[derive(Debug, PartialEq, Clone)]
pub enum Marker {
    SOI,
    APP0,
    DQT,
    SOF0,
    SOF2,
    DHT,
    SOS,
    DRI,
    RST(u8),
    EOI,
    COM,
}

#[derive(Debug)]
pub struct QuantTable {
    pub id: u8,
    pub precision: u8,
    pub values: [u16; 64],
}

#[derive(Debug)]
pub struct FrameHeader {
    pub precision: u8,
    pub height: u16,
    pub width: u16,
    pub components: u8,
    pub comp_info: Vec<ComponentInfo>,
}

#[derive(Debug)]
pub struct ComponentInfo {
    pub id: u8,
    pub sampling_h: u8,
    pub sampling_v: u8,
    pub qtable_id: u8,
}

#[derive(Debug)]
pub struct ScanHeader {
    pub components: u8,
    pub comp_select: Vec<ScanComponent>,
    pub spectral_start: u8,
    pub spectral_end: u8,
    pub approx: u8,
}

#[derive(Debug)]
pub struct ScanComponent {
    pub id: u8,
    pub dc_table: u8,
    pub ac_table: u8,
}

#[derive(Debug)]
pub struct JpegHeaderInfo {
    pub markers: Vec<Marker>,
    pub quant_tables: Vec<QuantTable>,
    pub frame: Option<FrameHeader>,
    pub scan: Option<ScanHeader>,
    pub restart_interval: u16,
}

fn read_u16(data: &[u8]) -> u16 {
    ((data[0] as u16) << 8) | (data[1] as u16)
}

fn read_marker(data: &[u8], pos: usize) -> Option<(Marker, usize)> {
    if pos + 1 >= data.len() {
        return None;
    }
    if data[pos] != 0xFF {
        return None;
    }
    let byte = data[pos + 1];
    if byte == 0x00 {
        return None;
    }
    let marker = match byte {
        0xD8 => Marker::SOI,
        0xE0 => Marker::APP0,
        0xDB => Marker::DQT,
        0xC0 => Marker::SOF0,
        0xC2 => Marker::SOF2,
        0xC4 => Marker::DHT,
        0xDA => Marker::SOS,
        0xDD => Marker::DRI,
        0xFE => Marker::COM,
        0xD9 => Marker::EOI,
        b if (0xD0..=0xD7).contains(&b) => Marker::RST(b - 0xD0),
        _ => return None,
    };
    Some((marker, pos + 2))
}

pub struct JpegParser<'a> {
    data: &'a [u8],
    pos: usize,
    info: JpegHeaderInfo,
}

impl<'a> JpegParser<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        JpegParser {
            data,
            pos: 0,
            info: JpegHeaderInfo {
                markers: Vec::new(),
                quant_tables: Vec::new(),
                frame: None,
                scan: None,
                restart_interval: 0,
            },
        }
    }

    pub fn parse(&mut self) -> Result<&JpegHeaderInfo, String> {
        if self.data.len() < 2 || self.data[0] != 0xFF || self.data[1] != 0xD8 {
            return Err("Not a valid JPEG: missing SOI marker".into());
        }
        self.info.markers.push(Marker::SOI);
        self.pos = 2;

        loop {
            if self.pos >= self.data.len() {
                break;
            }
            if self.data[self.pos] != 0xFF {
                return Err(format!("Expected marker at byte {}, got 0x{:02X}", self.pos, self.data[self.pos]));
            }
            let marker = read_marker(self.data, self.pos).ok_or_else(|| format!("Unknown marker at {}", self.pos))?;
            self.pos = marker.1;
            self.info.markers.push(marker.0.clone());

            match marker.0 {
                Marker::SOS => {
                    self.info.scan = Some(self.parse_sos()?);
                    break;
                }
                Marker::SOI | Marker::EOI | Marker::RST(_) => {}
                _ => {
                    if self.pos + 1 >= self.data.len() {
                        break;
                    }
                    let length = read_u16(&self.data[self.pos..]) as usize;
                    let segment_start = self.pos + 2;
                    self.parse_segment(&marker.0, &self.data[segment_start..segment_start + length - 2])?;
                    self.pos = segment_start + length - 2;
                }
            }
        }

        Ok(&self.info)
    }

    fn parse_segment(&mut self, marker: &Marker, data: &[u8]) -> Result<(), String> {
        match marker {
            Marker::DQT => self.parse_dqt(data),
            Marker::SOF0 | Marker::SOF2 => self.parse_sof(data),
            Marker::DHT => self.parse_dht(data),
            Marker::DRI => {
                if data.len() >= 2 {
                    self.info.restart_interval = read_u16(data);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn parse_dqt(&mut self, data: &[u8]) -> Result<(), String> {
        let mut pos = 0;
        while pos + 65 <= data.len() {
            let info = data[pos];
            let precision = info >> 4;
            let id = info & 0x0F;
            let mut values = [0u16; 64];
            if precision == 0 {
                for i in 0..64 {
                    let zigzag = ZIGZAG[i] as usize;
                    values[zigzag] = data[pos + 1 + i] as u16;
                }
                pos += 65;
            } else {
                for i in 0..64 {
                    let zigzag = ZIGZAG[i] as usize;
                    values[zigzag] = read_u16(&data[pos + 1 + i * 2..]);
                }
                pos += 129;
            }
            self.info.quant_tables.push(QuantTable { id, precision, values });
        }
        Ok(())
    }

    fn parse_sof(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 8 {
            return Err("SOF segment too short".into());
        }
        let precision = data[0];
        let height = read_u16(&data[1..]);
        let width = read_u16(&data[3..]);
        let components = data[5];
        let mut comp_info = Vec::new();
        for i in 0..components as usize {
            let off = 6 + i * 3;
            if off + 3 > data.len() {
                return Err("SOF component data truncated".into());
            }
            comp_info.push(ComponentInfo {
                id: data[off],
                sampling_h: data[off + 1] >> 4,
                sampling_v: data[off + 1] & 0x0F,
                qtable_id: data[off + 2],
            });
        }
        self.info.frame = Some(FrameHeader { precision, height, width, components, comp_info });
        Ok(())
    }

    fn parse_dht(&mut self, _data: &[u8]) -> Result<(), String> {
        Ok(())
    }

    fn parse_sos(&mut self) -> Result<ScanHeader, String> {
        if self.pos + 1 >= self.data.len() {
            return Err("SOS truncated".into());
        }
        let length = read_u16(&self.data[self.pos..]) as usize;
        let segment_start = self.pos + 2;
        if segment_start >= self.data.len() {
            return Err("SOS data truncated".into());
        }
        let data = &self.data[segment_start..];
        let components = data[0] as usize;
        let mut comp_select = Vec::new();
        for i in 0..components {
            let off = 1 + i * 2;
            if off + 1 >= data.len() || off + 1 >= length {
                break;
            }
            comp_select.push(ScanComponent {
                id: data[off],
                dc_table: data[off + 1] >> 4,
                ac_table: data[off + 1] & 0x0F,
            });
        }
        let spec_off = 1 + components * 2;
        Ok(ScanHeader {
            components: components as u8,
            comp_select,
            spectral_start: if spec_off < length && spec_off < data.len() { data[spec_off] } else { 0 },
            spectral_end: if spec_off + 1 < length && spec_off + 1 < data.len() { data[spec_off + 1] } else { 63 },
            approx: if spec_off + 2 < length && spec_off + 2 < data.len() { data[spec_off + 2] } else { 0 },
        })
    }
}

const ZIGZAG: [u8; 64] = [
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

pub fn parse_header(data: &[u8]) -> Result<super::JpegInfo, String> {
    let mut parser = JpegParser::new(data);
    let info = parser.parse()?;

    let frame = info.frame.as_ref().ok_or("No frame header found")?;

    let mut sampling_h = [1u8; 3];
    let mut sampling_v = [1u8; 3];
    let mut qtables = [[0u16; 64]; 4];

    for (i, comp) in frame.comp_info.iter().enumerate() {
        if i < 3 {
            sampling_h[i] = comp.sampling_h;
            sampling_v[i] = comp.sampling_v;
        }
    }
    for qt in &info.quant_tables {
        if (qt.id as usize) < 4 {
            qtables[qt.id as usize] = qt.values;
        }
    }

    Ok(super::JpegInfo {
        width: frame.width,
        height: frame.height,
        components: frame.components,
        sampling_h,
        sampling_v,
        qtables,
        huff_dc: None,
        huff_ac: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_jpeg() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xFF, 0xD8]);
        buf.extend_from_slice(&[0xFF, 0xDB, 0, 67, 0]);
        for i in 0..64 {
            buf.push((i + 1) as u8);
        }
        buf.extend_from_slice(&[0xFF, 0xC0, 0, 17, 8, 0, 16, 0, 16, 3]);
        buf.extend_from_slice(&[1, 0x22, 0]);
        buf.extend_from_slice(&[2, 0x11, 0]);
        buf.extend_from_slice(&[3, 0x11, 0]);
        buf.extend_from_slice(&[0xFF, 0xDA, 0, 8, 1, 0, 0, 0, 0x3F, 0]);
        buf
    }

    #[test]
    fn test_parse_minimal_jpeg() {
        let data = minimal_jpeg();
        let info = parse_header(&data).unwrap();
        assert_eq!(info.width, 16);
        assert_eq!(info.height, 16);
        assert_eq!(info.components, 3);
        assert_eq!(info.sampling_h[0], 2);
        assert_eq!(info.sampling_v[0], 2);
    }

    #[test]
    fn test_missing_soi() {
        let data = vec![0, 0];
        assert!(parse_header(&data).is_err());
    }

    #[test]
    fn test_read_marker_soi() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0];
        let result = read_marker(&data, 0);
        assert_eq!(result, Some((Marker::SOI, 2)));
    }
}
