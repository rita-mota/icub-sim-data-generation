# Legacy research snapshots

These files are preserved for historical reference and are not part of the
supported dataset-generation pipeline:

- `robot_skills_before_t1_smooth.py`: controller snapshot predating the smooth
  T1 integration;
- `task_5_alternate.py`: unused alternate T5 implementation;
- `render_recording_deprecated.py`: superseded offline renderer.

They are not imported by `main_collect.py`, are excluded from behavioral
validation, and may not match the current dataset format. Use the maintained
implementations in `core/`, `tasks/`, and `tools/` for new work.
