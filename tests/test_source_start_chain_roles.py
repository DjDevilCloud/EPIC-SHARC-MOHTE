"""Reusable initial-chain roles, neutral v3 migration and causal execution."""
import unittest
import torch
from tests import test_source_ownership as ownership_fixtures
from signature_bank import BoundedIdentityReadout,WordSpanState
from data import _build_window_samples_from_text

class SourceStartChainRoleTests(unittest.TestCase):
    def roles(self,t,text):
        words=WordSpanState()
        for token in t.encode(text):words.observe(token,t.construction_units[token])
        words.boundary()
        return list(BoundedIdentityReadout.source_start_chain_roles(words,t).values())

    def test_roles_share_across_unfamiliar_initials_and_fragmented_names(self):
        _,t,_=ownership_fixtures.SourceOwnershipTests().fixture()
        a=self.roles(t,'K. Zorvax\n')
        b=self.roles(t,'C. Zorvaxquethorin\n')
        self.assertEqual(a,b)
        self.assertIn('initial=True',a[0])
        self.assertIn('after_initial=True',a[1])
        self.assertIn('line_start=False',a[1])

    def test_line_break_and_ordinary_word_do_not_claim_initial_continuation(self):
        _,t,_=ownership_fixtures.SourceOwnershipTests().fixture()
        for text in ('K.\nZorvax\n','Nera. Zorvax\n'):
            roles=self.roles(t,text)
            self.assertIn('after_initial=False',roles[1])
        self.assertIn('line_start=True',self.roles(t,'K.\nZorvax\n')[1])

    def test_trained_v3_migration_is_neutral_with_unused_slot_zero(self):
        f=ownership_fixtures.SourceOwnershipTests();helper,t,m=f.fixture();m.cfg.identity_readout_source_start_features='context_roles_v3';head=f.enable(m,t)
        d=m.cfg.d_model
        with torch.no_grad():
            head.native_span_start.weight.normal_(0,.2)
            head.native_span_start.weight[:,2*d:3*d].zero_()
            head.native_source_cursor_scale.fill_(.8)
        s=_build_window_samples_from_text(t,'<BOI>K. Zorvax Reed\nWho is the author?<EOI><BOO>K. Zorvax Reed<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():
            before=m(s.input_ids[None],**helper.inputs(s)).logits
            m.cfg.identity_readout_source_start_features='context_roles_v4'
            torch.testing.assert_close(before,m(s.input_ids[None],**helper.inputs(s)).logits,atol=0,rtol=0)

    def test_active_v4_streams_reloads_and_has_finite_gradients(self):
        ownership_fixtures.SourceOwnershipTests().active_path('context_roles_v4',posterior=True)

    def alias_fixture(self,t,text):
        words=WordSpanState()
        for token in t.encode(text):words.observe(token,t.construction_units[token])
        words.boundary()
        entries=[(token,(),(),serial,()) for serial,_,word in words.records for token in word]
        eligible={entry[3] for entry,ok in zip(entries,words.source_candidate_mask(entries)) if ok}
        aliases=BoundedIdentityReadout.source_start_prefix_aliases(words,t,eligible)
        return [(t.decode(list(word)),serial in aliases) for serial,_,word in words.records]

    def test_prefix_aliases_preserve_protocol_shared_line_and_source_occurrence(self):
        _,t,_=ownership_fixtures.SourceOwnershipTests().fixture()
        result=self.alias_fixture(t,'Context\tTalvessan Reed Quorin\nTalvessan Reed\nQuorin\nWho is the author?<EOI>')
        starts=[alias for word,alias in result if word=='talvessan']
        self.assertEqual(starts,[False,True])

    def test_aliases_require_matching_separators_and_exclude_query_evidence(self):
        _,t,_=ownership_fixtures.SourceOwnershipTests().fixture()
        for text in ('Talvessan|Reed Quorin\nTalvessan Reed\nWho is the author?<EOI>',
                     'Talvessan Reed\nTalvessan Reed Quorin<EOI>'):
            self.assertFalse(any(alias for word,alias in self.alias_fixture(t,text) if word=='talvessan'))

    def test_v5_leaves_unrelated_numeric_scoring_identical_with_trained_columns(self):
        f=ownership_fixtures.SourceOwnershipTests();helper,t,m=f.fixture();m.cfg.identity_readout_source_start_features='context_roles_v3';head=f.enable(m,t)
        with torch.no_grad():head.native_span_start.weight.normal_(0,.2)
        s=_build_window_samples_from_text(t,'<BOI>The report lists 125-min intervals.\nWhat interval is listed?<EOI><BOO>125-min intervals<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():
            before=m(s.input_ids[None],**helper.inputs(s)).logits
            m.cfg.identity_readout_source_start_features='context_roles_v5'
            torch.testing.assert_close(before,m(s.input_ids[None],**helper.inputs(s)).logits,atol=0,rtol=0)

    def test_active_v5_streams_reloads_and_has_finite_gradients(self):
        ownership_fixtures.SourceOwnershipTests().active_path('context_roles_v5',posterior=True)

if __name__=='__main__':unittest.main()
