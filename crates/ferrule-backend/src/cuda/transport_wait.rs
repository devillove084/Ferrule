//! Host scheduling for synchronous exact-ticket waits.
//!
//! This module never observes or fabricates completion. The caller supplies the
//! exact-ticket query and remains responsible for custody and error handling.
pub(super) fn wait<T, E>(
    mut poll: impl FnMut() -> Result<Option<T>, E>,
    mut yield_now: impl FnMut(),
) -> Result<T, E> {
    loop {
        if let Some(completion) = poll()? {
            return Ok(completion);
        }
        // Keep the original low-latency query/yield loop at every iteration.
        // A yield is scheduling only: the next exact-ticket query is still required.
        yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn ready_and_error_paths_do_not_pause_or_retry() {
        assert_eq!(wait(|| Ok::<_, ()>(Some(7)), || panic!("yielded")), Ok(7));
        assert_eq!(
            wait(|| Err::<Option<()>, _>("query error"), || panic!("yielded")),
            Err("query error")
        );
    }

    #[test]
    fn default_policy_only_yields_even_after_long_pending_sequences() {
        for ready_at in [1, 2, 8, 9, 40, 4096] {
            let queries = Cell::new(0);
            let yields = Cell::new(0);
            let result = wait(
                || {
                    assert_eq!(queries.get(), yields.get());
                    queries.set(queries.get() + 1);
                    Ok::<_, ()>((queries.get() == ready_at).then_some("same ticket"))
                },
                || yields.set(yields.get() + 1),
            );
            assert_eq!(result, Ok("same ticket"));
            assert_eq!(queries.get(), ready_at);
            assert_eq!(yields.get(), ready_at - 1);
        }
    }

    #[test]
    fn yielding_does_not_turn_pending_into_completion() {
        let mut outcomes = [Ok(None), Ok(None), Ok(Some("exact proof"))].into_iter();
        let mut yields = 0;
        let result = wait::<_, ()>(
            || outcomes.next().expect("must stop at completion"),
            || yields += 1,
        );
        assert_eq!(result, Ok("exact proof"));
        assert_eq!(yields, 2);
    }

    #[test]
    fn query_error_keeps_custody_outside_the_wait_helper() {
        let custody = Cell::new(true);
        let queries = Cell::new(0u32);
        let result = wait(
            || {
                assert!(custody.get());
                queries.set(queries.get() + 1);
                if queries.get() == 12 {
                    Err("unknown")
                } else {
                    Ok(None::<()>)
                }
            },
            || assert!(custody.get()),
        );
        assert_eq!(result, Err("unknown"));
        assert_eq!(queries.get(), 12);
        assert!(custody.get());
    }
}
