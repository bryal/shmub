#![feature(duration_float)]

extern crate cpal;
extern crate shmub_common;
extern crate std_semaphore;

use cpal::{Format, OutputBuffer, SampleFormat, SampleRate, StreamData, UnknownTypeOutputBuffer};
use shmub_common::*;
use std::collections::VecDeque;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::thread;
use std_semaphore::Semaphore;
use std::time;

const PORT: u16 = 14320;
const BUFFER_LATENCY_MS: usize = 30;
const BUFFER_LATENCY_FRAMES: usize = (BUFFER_LATENCY_MS * SAMPLE_RATE as usize) / 1000;
const SEQ_RESTART_MARGIN: u32 = 100;

fn main() {
    println!("Shmub Audio Server");
    println!("Receive streamed UDP audio data and output to connected sound device.");
    println!(
        "Buffer size: {} frames; Buffer time: ~{}ms; Restart margin: {} frames",
        BUFFER_LATENCY_FRAMES, BUFFER_LATENCY_MS, SEQ_RESTART_MARGIN
    );
    let device = prompt_device();
    let format = Format {
        channels: N_CHANNELS as u16,
        sample_rate: SampleRate(SAMPLE_RATE),
        data_type: SampleFormat::I16,
    };
    let event_loop = cpal::EventLoop::new();
    let stream_id = event_loop
        .build_output_stream(&device, &format)
        .expect(&format!(
            "Failed to build output stream for device {} with format {:?}",
            device.name(),
            format
        ));
    event_loop.play_stream(stream_id);
    let socket = UdpSocket::bind((Ipv4Addr::new(0, 0, 0, 0), PORT)).expect("Failed to open socket");
    println!("Listening to port {}", PORT);

    let buffer = AudioBuffer::new(BUFFER_LATENCY_FRAMES);
    let buffer2 = buffer.clone();

    thread::spawn(move || {
        let mut prev_frame = [0, 0];
        event_loop.run(move |_, data| {
            let mut out_buf = get_i16_buffer(as_output_buffer(data));
            for out_frame in out_buf.chunks_mut(N_CHANNELS) {
                let frame = buffer2.try_pop_front().unwrap_or(prev_frame);
                for (out, &value) in out_frame.iter_mut().zip(&frame[..]) {
                    *out = value;
                }
                prev_frame = frame;
            }
            buffer2.wait_for_buffering_if_empty();
        });
    });

    let mut frames_per_s = 0;
    let mut t0 = time::Instant::now();
    let mut last_seq_index = 0;
    loop {
        let (packet, origin) = recv_packet(&socket);
        let is_newer = packet.seq_index > last_seq_index;
        let probably_reset = packet.seq_index < last_seq_index.saturating_sub(SEQ_RESTART_MARGIN);
        if is_newer || probably_reset {
            if !packet.seq_index == last_seq_index + 1 {
                println!(
                    "indirect successor. packet: {}, last: {}",
                    packet.seq_index, last_seq_index
                )
            }
            if probably_reset {
                println!(
                    "probably reset. packet: {}, last: {}",
                    packet.seq_index, last_seq_index
                );
            }
            for frame in packet.frames[..].iter().cloned() {
                frames_per_s += 1;
                buffer.try_push_back(frame);
            }
            last_seq_index = packet.seq_index;
            let t1 = time::Instant::now();
            let t = t1.duration_since(t0).as_float_secs();
            if t >= 5.0 {
                t0 = t1;
                println!("Frames per second: {}", frames_per_s as f64 / t);
                frames_per_s = 0;
            }
        } else {
            println!(
                "out of order packet. this: {}, last: {}",
                packet.seq_index, last_seq_index
            );
        }
    }
}

/// A thread-safe buffer of frames that handles buffering in a way that makes sense for audio
#[derive(Clone)]
struct AudioBuffer {
    guarded_queue: Arc<Mutex<VecDeque<Frame>>>,
    optimal_size: usize,
    available_frames: Arc<Semaphore>,
}

impl AudioBuffer {
    fn new(optimal_size: usize) -> Self {
        println!(
            "Creating audio buffer with optimal size of {}",
            optimal_size
        );
        assert!(optimal_size > 1, "Optimal size must be greater than 1");
        AudioBuffer {
            guarded_queue: Arc::new(Mutex::new(VecDeque::with_capacity(optimal_size * 2))),
            optimal_size,
            available_frames: Arc::new(Semaphore::new(0)),
        }
    }

    /// If the buffer is not unbuffering to `optimal_size`, push the frame
    ///
    /// Returns whether the frame was pushed
    fn try_push_back(&self, frame: Frame) -> bool {
        let mut queue = self.guarded_queue
            .lock()
            .expect("Error locking queue mutex");
        if queue.len() < self.optimal_size * 2 {
            queue.push_back(frame);
            self.available_frames.release();
            true
        } else {
            false
        }
    }

    fn wait_for_buffering(&self) {
        println!("buffering...");
        for _ in 0..self.optimal_size {
            self.available_frames.acquire();
        }
        for _ in 0..self.optimal_size {
            self.available_frames.release();
        }
    }

    fn wait_for_buffering_if_empty(&self) {
        let wait = {
            let queue = self.guarded_queue
                .lock()
                .expect("error locking queue mutex");
            let wait = queue.len() <= 10;
            wait
        };
        if wait {
            self.wait_for_buffering()
        }
    }

    /// If the buffer is not buffering to `optimal_size`, pop a frame
    fn pop_front(&self) -> Frame {
        self.available_frames.acquire();
        let (frame, wait) = {
            let mut queue = self.guarded_queue
                .lock()
                .expect("error locking queue mutex");
            let frame = queue.pop_front().expect("no element to pop queue");
            let wait = queue.len() <= 1;
            (frame, wait)
        };
        if wait {
            self.wait_for_buffering()
        }
        frame
    }

    fn try_pop_front(&self) -> Option<Frame> {
        let mut queue = self.guarded_queue
            .lock()
            .expect("error locking queue mutex");
        if queue.len() > 0 {
            self.available_frames.acquire();
            queue.pop_front()
        } else {
            None
        }
    }
}

fn get_i16_buffer<'b>(unknown_buf: UnknownTypeOutputBuffer<'b>) -> OutputBuffer<'b, i16> {
    match unknown_buf {
        UnknownTypeOutputBuffer::I16(buf) => buf,
        _ => panic!("Expected I16 buffer"),
    }
}

fn as_output_buffer(data: StreamData) -> UnknownTypeOutputBuffer {
    match data {
        StreamData::Output { buffer } => buffer,
        _ => panic!("Expected output data buffer"),
    }
}

fn recv_packet(socket: &UdpSocket) -> (Packet, SocketAddr) {
    let mut buf = [0; PACKET_SIZE];
    let (_, origin) = socket
        .recv_from(&mut buf[..])
        .expect("Error receiving from socket");
    (
        Packet::parse(&buf[..]).expect("Error parsing packet"),
        origin,
    )
}

fn prompt_device() -> cpal::Device {
    println!("Select audio output device:");
    let mut devices = cpal::output_devices().collect::<Vec<_>>();
    for (i, device) in devices.iter().enumerate() {
        println!(" {}) {}", i, device.name());
    }
    let line = prompt_line();
    let i = line.parse::<usize>().expect(&format!(
        "Failed to parse input as device index: line = \"{}\"",
        line
    ));
    if i >= devices.len() {
        panic!("Undefined device index");
    } else {
        devices.remove(i)
    }
}
