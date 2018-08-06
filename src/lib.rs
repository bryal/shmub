extern crate byteorder;

use byteorder::{ByteOrder, LittleEndian};
use std::io::{self, BufRead, Write};
use std::mem::{size_of, transmute};

macro_rules! print_flush {
    ($s: expr) => {
        print_flush!($s,);
    };
    ($format_str: expr, $($args: expr),*) => {
        print!($format_str, $($args),*);
        io::stdout().flush().unwrap();
    };
}

pub const N_CHANNELS: usize = 2;
pub const PACKET_PCM_SAMPLES_SIZE: usize = 512;
pub const PACKET_SIZE: usize = PACKET_PCM_SAMPLES_SIZE;
pub const PCM_SAMPLE_SIZE: usize = size_of::<[i16; N_CHANNELS]>();
pub const PACKET_N_PCM_SAMPLES: usize = PACKET_PCM_SAMPLES_SIZE / PCM_SAMPLE_SIZE;
pub const SAMPLE_RATE: u32 = 48000;

// Packet format:
// | SAMPLES |
// SAMPLES: 512 bytes of 128 pairs of le i16 pcm samples with sample rate of 48khz.
//          512 bytes seems like a good size to work well with UDP.

pub struct Packet {
    samples: [[i16; N_CHANNELS]; PACKET_N_PCM_SAMPLES],
}

impl Packet {
    pub fn new(samples: &[[i16; N_CHANNELS]]) -> Result<Packet, ()> {
        if samples.len() != PACKET_N_PCM_SAMPLES {
            Err(())
        } else {
            let mut array = [[0; N_CHANNELS]; PACKET_N_PCM_SAMPLES];
            array.copy_from_slice(samples);
            Ok(Packet { samples: array })
        }
    }

    pub fn samples(&self) -> &[[i16; N_CHANNELS]; PACKET_N_PCM_SAMPLES] {
        &self.samples
    }

    pub fn to_bytes(&self) -> [u8; PACKET_SIZE] {
        let mut bytes = [0; PACKET_SIZE];
        let samples = sample_pairs_as_singles(&self.samples);
        LittleEndian::write_i16_into(&samples[..], &mut bytes);
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
            let mut samples = [0i16; PACKET_N_PCM_SAMPLES * N_CHANNELS];
            LittleEndian::read_i16_into(buf, &mut samples[..]);
            Ok(Packet {
                samples: sample_singles_to_pairs(samples),
            })
        }
    }
}

fn sample_pairs_as_singles(
    samples: &[[i16; N_CHANNELS]; PACKET_N_PCM_SAMPLES],
) -> &[i16; PACKET_N_PCM_SAMPLES * N_CHANNELS] {
    // NOTE: I THINK this is portable?
    unsafe { transmute(samples) }
}

fn sample_singles_to_pairs(
    samples: [i16; PACKET_N_PCM_SAMPLES * N_CHANNELS],
) -> [[i16; N_CHANNELS]; PACKET_N_PCM_SAMPLES] {
    // NOTE: I THINK this is portable?
    unsafe { transmute(samples) }
}

pub fn prompt_line() -> String {
    print_flush!("> ");
    read_line()
}

fn read_line() -> String {
    let stdin = io::stdin();
    let s = stdin
        .lock()
        .lines()
        .next()
        .unwrap()
        .expect("Failed to read line from stdin");
    s
}
