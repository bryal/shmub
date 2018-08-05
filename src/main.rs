extern crate hound;
extern crate shmub_common;

use shmub_common::*;
use std::net::{Ipv4Addr, UdpSocket};

const PORT: u16 = 14320;

fn main() {
    let socket = UdpSocket::bind((Ipv4Addr::new(0, 0, 0, 0), PORT)).expect("Failed to open socket");
    let mut buf = [0; PACKET_SIZE];
    let mut packets = Vec::new();
    for _ in 0..4000 {
        socket
            .recv_from(&mut buf[..])
            .expect("Error receiving from socket");
        packets.push(Packet::parse(&buf[..]).expect("Error parsing packet"));
    }
    let pcm_samples = packets
        .into_iter()
        .flat_map(|p| p.samples()[..].iter().cloned().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let spec = hound::WavSpec {
        channels: N_CHANNELS as u16,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("recording.wav", spec).unwrap();
    for sample in pcm_samples {
        writer.write_sample(sample[0]).unwrap();
        writer.write_sample(sample[1]).unwrap();
    }
    writer.finalize().unwrap();
}
