# Lab 3: Simulate and fix DetachedInstanceError

## Reproduce pattern
This error usually appears when an ORM object is used after session close.

## Bad pattern (example)
```python
job = get_job(db, job_id)
db.close()
print(job.user.username)  # can raise DetachedInstanceError
```

## Fix pattern
Extract primitive values before session close, or re-query in a fresh session.

```diff
- job = get_job(db, job_id)
- db.close()
- name = job.user.username
+ job = get_job(db, job_id)
+ username = job.user.username if job and job.user else None
+ db.close()
+ name = username
```

## Verify
- Re-run action path and confirm no detached error in logs.
