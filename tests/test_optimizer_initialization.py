import copy
import unittest

import torch

from model import PrismalWaveModel
from train import (_build_optimizer,_assert_optimizer_parameter_coverage,
                   _optimizer_parameter_manifest,train_model)
from native_qa_transfer_study import generation_fields
from muon_optim import OptimizerStateVersionError


class OptimizerInitializationTests(unittest.TestCase):
    def fresh(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        _,tokenizer,model=RuntimeSignatureTests().setup_model()
        return model,tokenizer

    def test_enabled_lattice_is_registered_before_first_forward_for_each_optimizer(self):
        for name in ('adamw','muon','hierarchical'):
            with self.subTest(optimizer=name):
                model,t=self.fresh();self.assertNotIn('signature_lattice_attention',model._modules)
                optimizer=_build_optimizer(model,optimizer_name=name,base_lr=.001,cfg=model.cfg)
                self.assertIn('signature_lattice_attention',model._modules)
                _assert_optimizer_parameter_coverage(model,optimizer)
                ids,tracks=generation_fields(t,dict(context='A red cat.',question='What color is the cat?',answer='red'))
                weight=model.signature_lattice_attention.v_proj.weight
                before=weight.detach().clone()
                output=model(ids,**tracks,collect_telemetry=False)
                _assert_optimizer_parameter_coverage(model,optimizer)
                output.logits.square().mean().backward();self.assertIsNotNone(weight.grad)
                optimizer.step()
                self.assertFalse(torch.equal(before,weight))
                manifest=_optimizer_parameter_manifest(model,optimizer)
                self.assertTrue(any('signature_lattice_attention.v_proj.weight' in g['names'] for g in manifest['groups']))

    def test_lazy_modules_inherit_owner_dtype_and_eval_mode(self):
        model,t=self.fresh();model.double().eval()
        model.materialize_enabled_modules()
        self.assertFalse(model.signature_lattice_attention.training)
        self.assertEqual(model.signature_lattice_attention.q_proj.weight.dtype,torch.float64)
        self.assertFalse(model.unmaterialized_parameter_modules())

    def test_token_memory_is_materialized_when_enabled(self):
        model,t=self.fresh();cfg=copy.deepcopy(model.cfg);cfg.use_token_memory_cross_attention=True
        model=PrismalWaveModel(cfg);model._prismal_tokenizer=t;model.prepare_capacity_for_tokenizer(t)
        optimizer=_build_optimizer(model,optimizer_name='adamw',base_lr=.001,cfg=cfg)
        self.assertIn('token_memory_attention',model._modules)
        _assert_optimizer_parameter_coverage(model,optimizer)

    def test_partial_freeze_must_follow_materialization(self):
        model,t=self.fresh();model.construction_head.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError,'Materialize'):
            _build_optimizer(model,optimizer_name='adamw',base_lr=.001,cfg=model.cfg)
        model.materialize_enabled_modules()
        optimizer=_build_optimizer(model,optimizer_name='adamw',base_lr=.001,cfg=model.cfg)
        _assert_optimizer_parameter_coverage(model,optimizer)
        registered={id(p) for g in optimizer.param_groups for p in g['params']}
        self.assertTrue(all(id(p) not in registered for p in model.construction_head.parameters()))

    def test_coverage_rejects_missing_duplicate_and_replaced_parameters(self):
        for defect in ('missing','duplicate','stale'):
            model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters())
            if defect=='missing':model.extra=torch.nn.Parameter(torch.zeros(1))
            elif defect=='duplicate':optimizer.param_groups[0]['params'].append(model.weight)
            else:model.weight=torch.nn.Parameter(torch.ones(2,2))
            with self.subTest(defect=defect),self.assertRaisesRegex(RuntimeError,'coverage failed'):
                _assert_optimizer_parameter_coverage(model,optimizer)

    def test_position_growth_preserves_frozen_policy_and_dtype(self):
        model,t=self.fresh();model.position_embedding.double().requires_grad_(False)
        before=model.position_embedding.embedding.weight.detach().clone()
        model.position_embedding.ensure_capacity(len(before)+7)
        after=model.position_embedding.embedding.weight
        self.assertFalse(after.requires_grad)
        self.assertEqual(after.dtype,torch.float64)
        torch.testing.assert_close(before,after[:len(before)],rtol=0,atol=0)

    def test_saved_config_preserves_effective_nested_settings(self):
        import tempfile
        from pathlib import Path
        from train import save_checkpoint,load_bundle_from_checkpoint
        model,t=self.fresh();cfg=copy.deepcopy(model.cfg)
        cfg.use_nested_learning=True;cfg.nested_learning_mid_interval=3;cfg.nested_learning_mid_ema_beta=.7
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(model,folder,tokenizer=t,config=cfg)
            _,_,loaded=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False)
            self.assertTrue(loaded.use_nested_learning)
            self.assertEqual(loaded.nested_learning_mid_interval,3)
            self.assertEqual(loaded.nested_learning_mid_ema_beta,.7)

    def test_main_trainer_does_not_swallow_legacy_hierarchical_state_error(self):
        model,t=self.fresh();optimizer=_build_optimizer(model,optimizer_name='hierarchical',base_lr=.001,cfg=model.cfg)
        old=optimizer.state_dict();old.pop('precision_adaptive_schema_version')
        model._prismal_training_state=dict(optimizer_name='hierarchical',optimizer_state=old,global_step=10)
        with self.assertRaises(OptimizerStateVersionError):
            train_model(model,[],torch.device('cpu'),cfg=model.cfg,optimizer_name='hierarchical',steps=1,lr=.001,progress=False,use_amp=False)

    def test_same_optimizer_membership_mismatch_cannot_silently_reset_moments(self):
        model,t=self.fresh();optimizer=_build_optimizer(model,optimizer_name='adamw',base_lr=.001,cfg=model.cfg)
        saved=optimizer.state_dict();saved['param_groups'][0]['params'].pop()
        model._prismal_training_state=dict(optimizer_name='adamw',optimizer_state=saved,global_step=10)
        with self.assertRaisesRegex(OptimizerStateVersionError,'fresh optimizer'):
            train_model(model,[],torch.device('cpu'),cfg=model.cfg,optimizer_name='adamw',steps=1,lr=.001,progress=False,use_amp=False)

    def test_main_trainer_saves_and_resumes_a_partial_nested_window(self):
        import tempfile
        from pathlib import Path
        from torch.utils.data import DataLoader
        from data import TextWindowDataset,build_collate_fn
        from train import save_checkpoint,load_bundle_from_checkpoint
        model,t=self.fresh();cfg=copy.deepcopy(model.cfg)
        cfg.use_nested_learning=True;cfg.freeze_registry_observations=True
        cfg.nested_learning_local_interval=cfg.nested_learning_mid_interval=cfg.nested_learning_global_interval=2
        ds=TextWindowDataset(['red cat red cat.'],t,seq_len=64,max_samples=0)
        loader=DataLoader(ds,batch_size=1,collate_fn=build_collate_fn(t.pad_id,t.signature_pad_id))
        options=dict(cfg=cfg,optimizer_name='hierarchical',steps=1,lr=.001,progress=False,use_amp=False)
        train_model(model,loader,torch.device('cpu'),**options)
        self.assertTrue(model.cfg.use_nested_learning)
        first=model._prismal_training_state
        self.assertEqual(first['optimizer_state']['precision_adaptive_schema_version'],2)
        self.assertTrue(all(g['_nested_group_step']==1 for g in first['optimizer_state']['param_groups']))
        self.assertIn('optimizer_manifest',first)
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(model,folder,tokenizer=t)
            loaded,_,_=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu')
            train_model(loaded,loader,torch.device('cpu'),**options)
        second=loaded._prismal_training_state
        self.assertEqual(second['global_step'],2)
        self.assertTrue(all(g['_nested_group_step']==2 for g in second['optimizer_state']['param_groups']))
        self.assertTrue(any(v.get('step')==1 for v in second['optimizer_state']['state'].values()))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
