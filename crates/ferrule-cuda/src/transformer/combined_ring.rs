//! Conversion between combined ring-slot indices and logical paged indices.

use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CombinedRingTopkLayout {
    pub rows: usize,
    pub topk: usize,
    pub start_position: usize,
    pub position_stride: usize,
    pub window_size: usize,
}

impl CombinedRingTopkLayout {
    pub fn elements(&self) -> Result<usize> {
        self.rows
            .checked_mul(self.topk)
            .ok_or_else(|| Error::Internal("combined ring top-k element count overflow".into()))
    }

    pub fn validate(&self) -> Result<()> {
        if self.rows == 0 || self.topk == 0 || self.window_size == 0 {
            return Err(Error::Internal(format!(
                "invalid combined ring top-k layout: rows={} topk={} window_size={}",
                self.rows, self.topk, self.window_size
            )));
        }
        let last_position = self.position(self.rows - 1)?;
        if last_position > i32::MAX as usize {
            return Err(Error::Internal(format!(
                "combined ring logical position exceeds i32 output ABI: {last_position}"
            )));
        }
        for (field, value) in [
            ("rows", self.rows),
            ("topk", self.topk),
            ("start_position", self.start_position),
            ("position_stride", self.position_stride),
            ("window_size", self.window_size),
        ] {
            u32::try_from(value).map_err(|_| {
                Error::Internal(format!(
                    "combined ring top-k {field} exceeds u32 kernel ABI: {value}"
                ))
            })?;
        }
        let elements = self.elements()?;
        u32::try_from(elements).map_err(|_| {
            Error::Internal(format!(
                "combined ring top-k elements exceed u32 kernel ABI: {elements}"
            ))
        })?;
        Ok(())
    }

    pub fn position(&self, row: usize) -> Result<usize> {
        if row >= self.rows {
            return Err(Error::Internal(format!(
                "combined ring row {row} exceeds row count {}",
                self.rows
            )));
        }
        self.start_position
            .checked_add(
                row.checked_mul(self.position_stride)
                    .ok_or_else(|| Error::Internal("combined ring row position overflow".into()))?,
            )
            .ok_or_else(|| Error::Internal("combined ring absolute position overflow".into()))
    }

    pub fn derived_window_len(&self, row: usize) -> Result<usize> {
        Ok(self
            .position(row)?
            .checked_add(1)
            .unwrap_or(usize::MAX)
            .min(self.window_size))
    }

    /// Returns `(logical_index, plane_selector)`, or `(-1, -1)` when invalid.
    pub fn resolve(
        &self,
        row: usize,
        combined_index: i32,
        valid_window_len: usize,
    ) -> Result<(i32, i32)> {
        let position = self.position(row)?;
        if combined_index < 0 {
            return Ok((-1, -1));
        }
        let combined_index = combined_index as usize;
        if combined_index >= self.window_size {
            let compressed = combined_index - self.window_size;
            return Ok(match i32::try_from(compressed) {
                Ok(logical) => (logical, 1),
                Err(_) => (-1, -1),
            });
        }
        let maximum_visible = position
            .checked_add(1)
            .unwrap_or(usize::MAX)
            .min(self.window_size);
        if valid_window_len > maximum_visible {
            return Ok((-1, -1));
        }
        let slot = combined_index;
        let logical = if position < self.window_size {
            if slot >= valid_window_len || slot > position {
                return Ok((-1, -1));
            }
            slot
        } else {
            let current_slot = position % self.window_size;
            let age = (current_slot + self.window_size - slot) % self.window_size;
            if age >= valid_window_len {
                return Ok((-1, -1));
            }
            position - age
        };
        Ok((logical as i32, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(start_position: usize, rows: usize, stride: usize) -> CombinedRingTopkLayout {
        CombinedRingTopkLayout {
            rows,
            topk: 4,
            start_position,
            position_stride: stride,
            window_size: 4,
        }
    }

    #[test]
    fn unfilled_and_just_filled_slots_are_logical_positions() {
        let unfilled = layout(2, 1, 1);
        assert_eq!(unfilled.resolve(0, 0, 3).unwrap(), (0, 0));
        assert_eq!(unfilled.resolve(0, 2, 3).unwrap(), (2, 0));
        assert_eq!(unfilled.resolve(0, 3, 3).unwrap(), (-1, -1));

        let full = layout(3, 1, 1);
        for slot in 0..4 {
            assert_eq!(full.resolve(0, slot, 4).unwrap(), (slot, 0));
        }
    }

    #[test]
    fn wrapped_slots_map_to_absolute_positions_across_multiple_turns() {
        let once = layout(4, 1, 1);
        assert_eq!(once.resolve(0, 1, 4).unwrap(), (1, 0));
        assert_eq!(once.resolve(0, 0, 4).unwrap(), (4, 0));
        assert_eq!(once.resolve(0, 2, 4).unwrap(), (2, 0));

        let many = layout(10, 1, 1);
        assert_eq!(many.resolve(0, 3, 4).unwrap(), (7, 0));
        assert_eq!(many.resolve(0, 0, 4).unwrap(), (8, 0));
        assert_eq!(many.resolve(0, 2, 4).unwrap(), (10, 0));
        assert_eq!(many.resolve(0, 3, 2).unwrap(), (-1, -1));
    }

    #[test]
    fn rows_stride_and_compressed_indices_are_supported() {
        let layout = layout(5, 3, 2);
        assert_eq!(layout.position(0).unwrap(), 5);
        assert_eq!(layout.position(1).unwrap(), 7);
        assert_eq!(layout.position(2).unwrap(), 9);
        assert_eq!(layout.resolve(1, 3, 4).unwrap(), (7, 0));
        assert_eq!(layout.resolve(2, 6, 4).unwrap(), (2, 1));
        assert_eq!(layout.resolve(2, -1, 4).unwrap(), (-1, -1));
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn cuda_conversion_matches_cpu_for_batched_rows_and_wraps() {
        use crate::context::{CombinedRingWindowLens, CudaArtifactOperatorContext};

        let context = CudaArtifactOperatorContext::new().unwrap();
        let layout = CombinedRingTopkLayout {
            rows: 4,
            topk: 4,
            start_position: 2,
            position_stride: 3,
            window_size: 4,
        };
        let combined = vec![0, 2, 3, 4, 2, 1, 0, 6, 0, 3, -1, 5, 1, 0, 3, 7];
        let mut expected_logical = Vec::with_capacity(combined.len());
        let mut expected_selectors = Vec::with_capacity(combined.len());
        for row in 0..layout.rows {
            let window_len = layout.derived_window_len(row).unwrap();
            for item in 0..layout.topk {
                let (logical, selector) = layout
                    .resolve(row, combined[row * layout.topk + item], window_len)
                    .unwrap();
                expected_logical.push(logical);
                expected_selectors.push(selector);
            }
        }

        let combined_device = context.upload_i32_buffer(&combined).unwrap();
        let mut logical_device = context.zero_i32_buffer(combined.len()).unwrap();
        let mut selector_device = context.zero_i32_buffer(combined.len()).unwrap();
        context
            .convert_combined_ring_topk_indices_into(
                &combined_device,
                CombinedRingWindowLens::PositionDerived,
                layout,
                &mut logical_device,
                &mut selector_device,
            )
            .unwrap();
        assert_eq!(
            context.download_i32_buffer(&logical_device).unwrap(),
            expected_logical
        );
        assert_eq!(
            context.download_i32_buffer(&selector_device).unwrap(),
            expected_selectors
        );
    }
}
