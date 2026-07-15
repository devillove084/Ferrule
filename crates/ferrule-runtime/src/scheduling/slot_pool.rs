//! Fixed-capacity logical sequence-slot allocation.

use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvHandle(pub usize);

pub trait SequenceSlotPool {
    fn alloc_slot(&mut self) -> Result<KvHandle>;
    fn free_slot(&mut self, handle: KvHandle) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct FixedSequenceSlotPool {
    free_slots: Vec<usize>,
    allocated: Vec<bool>,
}

impl FixedSequenceSlotPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            free_slots: (0..capacity).rev().collect(),
            allocated: vec![false; capacity],
        }
    }

    pub fn capacity(&self) -> usize {
        self.allocated.len()
    }

    pub fn active_count(&self) -> usize {
        self.capacity() - self.free_slots.len()
    }

    pub fn available(&self) -> usize {
        self.free_slots.len()
    }
}

impl SequenceSlotPool for FixedSequenceSlotPool {
    fn alloc_slot(&mut self) -> Result<KvHandle> {
        let slot = self
            .free_slots
            .pop()
            .ok_or_else(|| Error::Internal("no free sequence slots".into()))?;
        self.allocated[slot] = true;
        Ok(KvHandle(slot))
    }

    fn free_slot(&mut self, handle: KvHandle) -> Result<()> {
        let Some(allocated) = self.allocated.get_mut(handle.0) else {
            return Err(Error::Internal(format!(
                "sequence slot {} out of range",
                handle.0
            )));
        };
        if !*allocated {
            return Err(Error::Internal(format!(
                "sequence slot {} is not allocated",
                handle.0
            )));
        }
        *allocated = false;
        self.free_slots.push(handle.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_pool_allocates_frees_and_reuses() {
        let mut pool = FixedSequenceSlotPool::new(2);
        let first = pool.alloc_slot().unwrap();
        let second = pool.alloc_slot().unwrap();
        assert_eq!(first, KvHandle(0));
        assert_eq!(second, KvHandle(1));
        assert!(pool.alloc_slot().is_err());
        pool.free_slot(first).unwrap();
        assert_eq!(pool.alloc_slot().unwrap(), first);
    }

    #[test]
    fn fixed_pool_rejects_invalid_free() {
        let mut pool = FixedSequenceSlotPool::new(1);
        assert!(pool.free_slot(KvHandle(0)).is_err());
        assert!(pool.free_slot(KvHandle(1)).is_err());
    }
}
