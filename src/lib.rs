extern crate byteorder;

use byteorder::{ByteOrder, LittleEndian};
use std::mem::size_of;

pub const PACKET_PCM_SAMPLES_SIZE: usize = 512;
pub const PCM_SAMPLE_SIZE: usize = size_of::<i16>();
pub const PACKET_N_PCM_SAMPLES: usize = PACKET_PCM_SAMPLES_SIZE / PCM_SAMPLE_SIZE;
pub const PACKET_SIZE: usize = PACKET_PCM_SAMPLES_SIZE;

// Packet format:
// | SAMPLES |
// SAMPLES: 512 bytes of le i16 pcm samples.

pub struct Packet {
    // 512 bytes seems like a good size to work well with udp
    samples: [i16; PACKET_N_PCM_SAMPLES],
}

impl Packet {
    pub fn new(samples: &[i16]) -> Result<Packet, ()> {
        if samples.len() != PACKET_N_PCM_SAMPLES {
            Err(())
        } else {
            let mut array = [0; PACKET_N_PCM_SAMPLES];
            array.copy_from_slice(samples);
            Ok(Packet { samples: array })
        }
    }

    pub fn samples(&self) -> &[i16; PACKET_N_PCM_SAMPLES] {
        &self.samples
    }

    pub fn to_bytes(&self) -> [u8; PACKET_SIZE] {
        let mut bytes = [0; PACKET_SIZE];
        LittleEndian::write_i16_into(&self.samples[..], &mut bytes);
        bytes
    }

    pub fn parse(buf: &[u8]) -> Result<Packet, String> {
        if buf.len() != PACKET_SIZE {
            Err(format!(
                "Expected buffer of {} bytes, found {} bytes",
                PACKET_SIZE,
                buf.len()
            ))
        } else {
            let mut samples = [0; PACKET_N_PCM_SAMPLES];
            LittleEndian::read_i16_into(buf, &mut samples[..]);
            Ok(Packet { samples })
        }
    }
}
