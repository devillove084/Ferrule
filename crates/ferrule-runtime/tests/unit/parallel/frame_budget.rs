use super::*;

#[test]
fn charges_token_bucket_and_envelope_headers_in_both_directions() {
    let size = NumericFrameSize::new(2048, 256).unwrap();
    assert_eq!(size.payload_bytes(), 5_865_728);
    assert_eq!(size.frame_bytes(), 5_866_752);
    let limits = ProcessFrameLimits {
        max_command_bytes: size.payload_bytes(),
        max_output_bytes: size.payload_bytes(),
        max_frame_bytes: size.frame_bytes(),
        ..Default::default()
    };
    assert!(size.fits(limits));
    assert!(!size.fits(ProcessFrameLimits {
        max_command_bytes: size.payload_bytes() - 1,
        ..limits
    }));
    assert!(!size.fits(ProcessFrameLimits {
        max_output_bytes: size.payload_bytes() - 1,
        ..limits
    }));
    assert!(!size.fits(ProcessFrameLimits {
        max_frame_bytes: size.frame_bytes() - 1,
        ..limits
    }));
    let empty = NumericFrameSize::new(0, 0).unwrap();
    assert_eq!(empty.payload_bytes(), 256);
    assert_eq!(empty.frame_bytes(), 1280);
}

#[test]
fn every_arithmetic_stage_is_checked_without_a_representative_allocation() {
    for (hidden, tokens) in [
        (usize::MAX, 1),       // hidden * 11
        (usize::MAX / 11, 0),  // token header, even with zero tokens
        (1, usize::MAX),       // token count
        (0, usize::MAX / 384), // bucket header
    ] {
        assert_eq!(
            NumericFrameSize::new(hidden, tokens),
            Err(FrameSizeOverflow::Payload)
        );
    }
    let tokens = (usize::MAX - 256) / 384;
    assert_eq!(
        NumericFrameSize::new(0, tokens),
        Err(FrameSizeOverflow::Envelope)
    );
    // Successful enormous preflight also must not allocate a bucket/frame.
    let tokens = (usize::MAX - 1280) / 384;
    let size = NumericFrameSize::new(0, tokens).unwrap();
    assert_eq!(size.payload_bytes(), tokens * 384 + 256);
    assert_eq!(size.frame_bytes(), tokens * 384 + 1280);
}
