use ferrule_core::Result;

/// Single-sequence contiguous CPU KV state used by the FP32 reference path.
///
/// The current OLMoE CPU forward API still consumes raw K/V vectors. This type
/// makes ownership and position advancement explicit at the runtime boundary so
/// future CPU model families can reuse the same decode-state contract.
pub struct CpuContiguousKvState {
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pos: usize,
}

impl CpuContiguousKvState {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            v_cache: (0..num_layers).map(|_| Vec::new()).collect(),
            pos: 0,
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self, num_layers: usize) {
        *self = Self::new(num_layers);
    }

    pub fn decode_one<T>(
        &mut self,
        f: impl FnOnce(&mut Vec<Vec<f32>>, &mut Vec<Vec<f32>>, usize) -> Result<T>,
    ) -> Result<T> {
        let out = f(&mut self.k_cache, &mut self.v_cache, self.pos)?;
        self.pos += 1;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_one_advances_position() {
        let mut state = CpuContiguousKvState::new(2);
        assert_eq!(state.position(), 0);
        let out = state
            .decode_one(|k, v, pos| {
                assert_eq!(pos, 0);
                k[0].push(1.0);
                v[0].push(2.0);
                Ok(7)
            })
            .unwrap();
        assert_eq!(out, 7);
        assert_eq!(state.position(), 1);
    }

    #[test]
    fn reset_reinitializes_layers_and_position() {
        let mut state = CpuContiguousKvState::new(1);
        state
            .decode_one(|k, _, _| {
                k[0].push(1.0);
                Ok(())
            })
            .unwrap();
        state.reset(3);
        assert_eq!(state.position(), 0);
        state
            .decode_one(|k, v, pos| {
                assert_eq!(pos, 0);
                assert_eq!(k.len(), 3);
                assert_eq!(v.len(), 3);
                Ok(())
            })
            .unwrap();
    }
}
