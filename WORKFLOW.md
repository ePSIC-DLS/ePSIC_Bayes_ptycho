See WORKFLOW.md produced previously.

Campaign States:

evaluated
↓
proposed
↓
prepared
↓
submitted
↓
completed_pending_evaluation
↓
evaluated

terminal:

converged

Manual workflow:

doctor()
propose_next_batch()
prepare_proposed_batch()
submit_prepared_batch()
monitor_submitted_batch()
evaluate_completed_batch()

Automated workflow:

run_automated_loops()

Safe stop:

request_stop()