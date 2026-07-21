"""Bus subscribers: airc's read side of the suite's pub/sub topics.

Where watchers/ once polled the world directly, airc now consumes what the
watcher and processor components publish: commentary triages the raw commit
stream (topics/repo/*) and renders findings the review component confirms
(topics/chat/findings). Each subscriber owns a cursor and is idempotent;
delivery is at-least-once (ack after handle).
"""
