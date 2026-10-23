"""CPU-free concurrency regressions; run with unittest or pytest."""
import concurrent.futures
import unittest
from tinygpt.concurrency import OneAtATime


class GenerationSlotTests(unittest.TestCase):
    def test_only_one_simultaneous_owner(self):
        slot = OneAtATime()
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            owners = list(pool.map(lambda _: slot.try_acquire(), range(24)))
        accepted = [owner for owner in owners if owner is not None]
        self.assertEqual(len(accepted), 1)
        self.assertTrue(slot.release(accepted[0]))

    def test_old_cleanup_cannot_release_a_new_generation(self):
        slot = OneAtATime()
        old = slot.try_acquire()
        self.assertTrue(slot.release(old))
        current = slot.try_acquire()
        self.assertFalse(slot.release(old))
        self.assertIsNone(slot.try_acquire())
        self.assertTrue(slot.release(current))

    def test_only_the_owner_can_release_and_cleanup_is_idempotent(self):
        slot = OneAtATime()
        owner = slot.try_acquire()
        self.assertFalse(slot.release(None))
        self.assertFalse(slot.release(object()))
        self.assertIsNone(slot.try_acquire())
        self.assertTrue(slot.release(owner))
        self.assertFalse(slot.release(owner))
