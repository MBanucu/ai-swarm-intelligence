/// Flat binary trie node for branchless Huffman decoding.
/// `next` stores either a child index (bit 15 = 0) or a symbol (bit 15 = 1).
#[derive(Debug, Clone, Default)]
pub struct FlatTrie {
    pub nodes: Vec<u16>,   // bit 15 = leaf flag, bits 14-0 = symbol or child index
}

/// Standard Huffman table (used by `decode_huffman_block` with tree-walk fallback).
#[derive(Debug, Clone, Default)]
pub struct HuffmanTable {
    pub id: u8,
    pub class: u8,
    pub codes: Vec<(u16, u8)>,
    pub min_code: [i32; 16],
    pub max_code: [i32; 16],
    pub val_ptr: [i32; 16],
    pub values: Vec<u8>,
    pub flat_trie: Option<FlatTrie>,   // branchless trie (built lazily)
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
            flat_trie: None,
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

    /// Build the flat binary trie for branchless decoding.
    pub fn build_flat_trie(&mut self) {
        let mut nodes = Vec::new();
        nodes.push(0u16); // root
        let mut code: u32 = 0;

        for len in 1..=16 {
            let count = self.values_in_len(len);
            for _ in 0..count {
                // Walk the tree using the current code bits to find where to insert
                let mut node_idx = 0usize;
                for bit_pos in (0..len).rev() {
                    let bit = ((code >> bit_pos) & 1) as usize;
                    let entry = nodes[node_idx];
                    if entry & 0x8000 != 0 {
                        // Leaf where internal node should be — should not happen with valid Huffman
                        break;
                    }
                    let child_offset = entry as usize;
                    if child_offset == 0 {
                        // Need to create child
                        let new_idx = nodes.len();
                        if node_idx < nodes.len() {
                            nodes[node_idx] = new_idx as u16;
                        }
                        nodes.push(0);
                        nodes.push(0);
                        node_idx = new_idx + bit;
                    } else {
                        let child_idx = child_offset + bit;
                        if child_idx >= nodes.len() {
                            nodes.resize(child_idx + 1, 0);
                        }
                        node_idx = child_idx;
                    }
                }
                // Mark as leaf with symbol
                let sym = self.value_for_code(code, len as u8);
                if node_idx < nodes.len() {
                    nodes[node_idx] = 0x8000 | (sym as u16 & 0x7FFF);
                }
                code += 1;
            }
            code <<= 1;
        }

        self.flat_trie = Some(FlatTrie { nodes });
    }

    fn values_in_len(&self, len: usize) -> usize {
        if len == 0 { return 0; }
        let hi = self.val_ptr[len - 1];
        if hi < 0 { return 0; }
        let count = if len > 1 {
            let lo = self.val_ptr[len - 2];
            if lo < 0 { hi + 1 } else { hi - lo }
        } else {
            hi + 1
        };
        count as usize
    }

    fn value_for_code(&self, code: u32, len: u8) -> u8 {
        let idx = len as usize;
        if idx == 0 || idx > 16 { return 0; }
        let min = self.min_code[idx - 1];
        let code_i32 = code as i32;
        if code_i32 >= min && code_i32 <= self.max_code[idx - 1] {
            let index = self.val_ptr[idx - 1] + (code_i32 - min);
            if (index as usize) < self.values.len() {
                return self.values[index as usize];
            }
        }
        0
    }
}

/// Branchless Huffman symbol decode using a flat binary trie.
///
/// The trie is stored as a flat array: even indices = "0" child, odd = "1" child.
/// Bit 15 of an entry marks a leaf; the lower 15 bits contain the symbol.
/// Returns -1 on error.
fn decode_symbol_trie(data: &[u8], bit_offset: usize, trie: &[u16]) -> i32 {
    let mut state = 0u16;          // root node (index 0 = "0" child, index 1 = "1" child)
    let mut off = bit_offset;

    loop {
        let byte_idx = off >> 3;
        if byte_idx >= data.len() {
            return -1;
        }
        let bit = ((data[byte_idx] >> (7 - (off & 7))) & 1) as usize;
        off += 1;

        let child_idx = (state as usize) + bit;
        if child_idx >= trie.len() {
            return -1;
        }
        let entry = trie[child_idx];
        if entry & 0x8000 != 0 {
            // Leaf: symbol is in lower bits
            return (entry & 0x7FFF) as i32;
        }
        state = entry;
    }
}

pub fn decode_symbol(data: &[u8], bit_offset: usize, table: &HuffmanTable) -> i32 {
    // Use branchless trie if available
    if let Some(ref trie) = table.flat_trie {
        return decode_symbol_trie(data, bit_offset, &trie.nodes);
    }

    // Fallback to tree-walk
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
