#![feature(type_ascription)]

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
pub const PACKET_PCM_SAMPLES_SIZE: usize = 1200;
pub const PCM_SAMPLE_SIZE: usize = size_of::<Sample>();
pub const PACKET_N_PCM_SAMPLES: usize = PACKET_PCM_SAMPLES_SIZE / PCM_SAMPLE_SIZE;
pub const PACKET_N_PCM_FRAMES: usize = PACKET_N_PCM_SAMPLES / N_CHANNELS;
pub const SEQ_INDEX_SIZE: usize = size_of::<u32>();
pub const PACKET_SIZE: usize = SEQ_INDEX_SIZE + PACKET_PCM_SAMPLES_SIZE;
pub const SAMPLE_RATE: u32 = 44100;

pub type Sample = i16;
pub type Frame = [Sample; N_CHANNELS];

// Packet format:
// | INDEX | SAMPLES |
//
// INDEX: Sequence index / package number. UInt32 (4 bytes). Increased
//        by one for every package sent.  Can be used by server to not
//        play a packet that arrived behind a newer packet.
//
// FRAMES: 512 bytes of 128 pairs of le i16 pcm samples with sample
//         rate of 48khz. 512 bytes seems like a good size to work
//         well with UDP.

pub struct Packet {
    pub seq_index: u32,
    pub frames: [Frame; PACKET_N_PCM_FRAMES],
}

impl Packet {
    pub fn new(seq_index: u32, frames: &[Frame]) -> Result<Packet, ()> {
        if frames.len() != PACKET_N_PCM_FRAMES {
            Err(())
        } else {
            let mut array = [[0; N_CHANNELS]; PACKET_N_PCM_FRAMES];
            array.copy_from_slice(frames);
            Ok(Packet {
                seq_index,
                frames: array,
            })
        }
    }

    pub fn to_bytes(&self) -> [u8; PACKET_SIZE] {
        let mut bytes = [0; PACKET_SIZE];
        let mut i = 0;
        LittleEndian::write_u32(&mut bytes[i..], self.seq_index);
        i += SEQ_INDEX_SIZE;
        let samples = frames_as_samples(&self.frames);
        LittleEndian::write_i16_into(&samples[..], &mut bytes[i..]);
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
            let mut i = 0;
            let seq_index = LittleEndian::read_u32(&buf[i..]);
            i += SEQ_INDEX_SIZE;
            let mut samples = [0: Sample; PACKET_N_PCM_SAMPLES];
            LittleEndian::read_i16_into(&buf[i..], &mut samples[..]);
            Ok(Packet {
                seq_index,
                frames: samples_to_frames(samples),
            })
        }
    }
}

fn frames_as_samples(frames: &[Frame; PACKET_N_PCM_FRAMES]) -> &[Sample; PACKET_N_PCM_SAMPLES] {
    // NOTE: I THINK this is portable?
    unsafe { transmute(frames) }
}

fn samples_to_frames(samples: [i16; PACKET_N_PCM_SAMPLES]) -> [Frame; PACKET_N_PCM_FRAMES] {
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
