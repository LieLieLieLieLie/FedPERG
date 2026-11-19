"""Invariants required by the new attribution controls."""
import unittest

import torch

from fedperg.aggregators import PERGAggregator, BaselineAggregator


class RevisionControls(unittest.TestCase):
    def aggregator(self, variant='lite_paired_gate_bank_selector'):
        return PERGAggregator([torch.zeros(4)], 48, 4, .003, 1, .3,
                              torch.device('cpu'), variant)

    def test_permuted_preserves_weight_multisets_and_router_candidates(self):
        deltas = [[torch.tensor(v, dtype=torch.float)] for v in
                  ([1, 2, -3, 4], [-3, 4, 2, 1], [2, -1, 4, 3])]
        p, reward = torch.tensor([.2, .3, .5]), torch.tensor([.1, -.1, .2])
        a, b = self.aggregator(), self.aggregator()
        b.evidence_control, b.control_seed = 'permuted', 70
        x, y = a.aggregate(deltas, p, reward, 1), b.aggregate(deltas, p, reward, 1)
        self.assertEqual(len(x.expert_deltas), 7)
        self.assertEqual(len(y.expert_deltas), 7)
        torch.testing.assert_close(x.weights.sort(dim=0).values, y.weights.sort(dim=0).values)
        for i in (0, 1, 3, 5):
            torch.testing.assert_close(x.expert_deltas[i][0], y.expert_deltas[i][0])

    def test_router_evidence_duplicates(self):
        agg = self.aggregator('lite_paired_gate_bank_router_control')
        x = agg.aggregate([[torch.ones(4)], [-torch.ones(4)]],
                          torch.tensor([.4, .6]), torch.zeros(2), 1)
        for r, e in ((1, 2), (3, 4), (5, 6)):
            torch.testing.assert_close(x.expert_deltas[r][0], x.expert_deltas[e][0])

    def test_gate_cannot_recover_cancelled_coordinate(self):
        x = self.aggregator().aggregate([[torch.ones(4)], [-torch.ones(4)]],
             torch.tensor([.5, .5]), torch.zeros(2), 1)
        for candidate in x.expert_deltas:
            self.assertTrue(torch.equal(candidate[0], torch.zeros(4)))

    def test_fednova_and_loss_weighting_equations(self):
        deltas = [[torch.tensor([1.])], [torch.tensor([-1.])]]
        p, steps = torch.tensor([.5, .5]), torch.tensor([1., 2.])
        nova = BaselineAggregator('FedNova', [torch.zeros(1)], 0)
        out, _ = nova.aggregate(deltas, p, torch.zeros(2), steps, 1, torch.zeros(1))
        torch.testing.assert_close(out[0], torch.tensor([.375]))
        lbw = BaselineAggregator('FedLBW', [torch.zeros(1)], 0)
        _, weights = lbw.aggregate(deltas, p, torch.zeros(2), steps, 1,
                                   torch.zeros(1), torch.tensor([1., 3.]))
        torch.testing.assert_close(weights, torch.tensor([.75, .25]))


if __name__ == '__main__':
    unittest.main()
