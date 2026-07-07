#[derive(Debug, Clone, Default)]
pub struct HuffmanTable {
    pub id: u8,
    pub class: u8,
    pub codes: Vec<(u16, u8)>,
    pub min_code: [i32; 16],
    pub max_code: [i32; 16],
    pub val_ptr: [i32; 16],
    pub values: Vec<u8>,
}

impl HuffmanTable {
    pub fn from_dht(id: u8, class: u8, bits: &[u8; 16], values: &[u8]) -> Self {
        let mut table = HuffmanTable {
            id,
            class,
            min_code: [-1i32; 16],
            max_code: [-1i32; 16],
            val_ptr: [-1i32; 16],
            values: values.to_vec(),
            codes: Vec::new(),
        };
        table.build_tables(bits);
        table
    }

    fn build_tables(&mut self, bits: &[u8; 16]) {
        let mut code: i32 = 0;
        for i in 0..16 {
            if bits[i] > 0 {
                self.val_ptr[i] = code;
                self.min_code[i] = code;
                code += bits[i] as i32;
                self.max_code[i] = code - 1;
            } else {
                self.max_code[i] = -1;
            }
            code <<= 1;
        }
    }
}

pub fn decode_symbol(data: &[u8], bit_offset: usize, table: &HuffmanTable) -> i32 {
    let mut code: u32 = 0;
    let mut bits_read: u8 = 0;

    for i in 0..16usize {
        let byte_idx = (bit_offset + i) / 8;
        if byte_idx >= data.len() {
            return -1;
        }
        let bit = ((data[byte_idx] >> (7 - ((bit_offset + i) % 8))) & 1) as u32;
        code = (code << 1) | bit;
        bits_read += 1;

        let len = bits_read as usize;
        if len <= 16 && table.max_code[len - 1] >= 0 {
            let code_i32 = code as i32;
            if code_i32 >= table.min_code[len - 1] && code_i32 <= table.max_code[len - 1] {
                let index = table.val_ptr[len - 1] + (code_i32 - table.min_code[len - 1]);
                if (index as usize) < table.values.len() {
                    return table.values[index as usize] as i32;
                }
            }
        }
    }
    -1
}

pub fn decode_huffman_block(
    data: &[u8],
    bit_offset: &mut usize,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    prev_dc: &mut i32,
    qtable: &[u16; 64],
) -> [i32; 64] {
    let zigzag: [usize; 64] = [
        0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ];

    let mut block = [0i32; 64];

    let dc_cat = decode_symbol(data, *bit_offset, dc_table) as u32;
    *bit_offset += dc_cat as usize;

    if dc_cat > 0 {
        let mut dc_val: i32 = 0;
        for _ in 0..dc_cat {
            let byte = data[*bit_offset / 8];
            let bit = ((byte >> (7 - (*bit_offset % 8))) & 1) as i32;
            dc_val = (dc_val << 1) | bit;
            *bit_offset += 1;
        }
        if dc_val < (1 << (dc_cat - 1)) as i32 {
            dc_val -= (1 << dc_cat) as i32 - 1;
        }
        *prev_dc += dc_val;
    }
    block[0] = *prev_dc * qtable[0] as i32;

    let mut k = 1;
    while k < 64 {
        let rs = decode_symbol(data, *bit_offset, ac_table);
        *bit_offset += 1;
        if rs < 0 {
            break;
        }
        let r = (rs >> 4) & 0x0F;
        let s = rs & 0x0F;

        if s == 0 && r == 0 {
            break;
        }
        if s == 0 && r == 15 {
            k += 16;
            continue;
        }
        k += r as usize;
        if k >= 64 {
            break;
        }

        let mut ac_val: i32 = 0;
        for _ in 0..s {
            let byte = data[*bit_offset / 8];
            let bit = ((byte >> (7 - (*bit_offset % 8))) & 1) as i32;
            ac_val = (ac_val << 1) | bit;
            *bit_offset += 1;
        }
        if ac_val < (1 << (s - 1)) as i32 {
            ac_val -= (1 << s) as i32 - 1;
        }

        let zig = zigzag[k];
        block[zig] = ac_val * qtable[zig] as i32;
        k += 1;
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_table_basic() {
        let bits: [u8; 16] = [0, 2, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let values: Vec<u8> = vec![0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21];
        let table = HuffmanTable::from_dht(0, 0, &bits, &values);
        assert_eq!(table.values.len(), 9);
        assert!(table.max_code[1] >= 0);
    }

    #[test]
    fn test_decode_symbol_easy() {
        let bits: [u8; 16] = [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let values: Vec<u8> = vec![0x00, 0x01];
        let table = HuffmanTable::from_dht(0, 0, &bits, &values);
        let data = [0b00000000u8];
        let sym = decode_symbol(&data, 0, &table);
        assert_eq!(sym, 0);
    }
}
