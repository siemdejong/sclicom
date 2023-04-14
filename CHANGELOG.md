# Changelog

<!--next-version-placeholder-->

## v4.10.1 (2023-04-14)
### Fix
* Change defaults ([`7240d9e`](https://github.com/siemdejong/dpat/commit/7240d9e93fa030cd79eb0c1534931e1ac2520f70))

## v4.10.0 (2023-04-13)
### Feature
* Allow for CCMIL hparam tuning ([`9e587f8`](https://github.com/siemdejong/dpat/commit/9e587f8765a36bc5e68525c567436ba922eeda31))

### Fix
* Typing of example array ([`cb98f39`](https://github.com/siemdejong/dpat/commit/cb98f397736e6860e3427cc0ad92bbca303b0387))
* Make random tensor as example ([`a8d21f0`](https://github.com/siemdejong/dpat/commit/a8d21f0440708b20894a659864a582f82328e072))
* Move tokenized inputs to device ([`2b75731`](https://github.com/siemdejong/dpat/commit/2b75731c7e4dc5ac8e93ab322c7a8f0d039da54e))
* Make new example input for lightning ([`a431ab2`](https://github.com/siemdejong/dpat/commit/a431ab2f82f2977900612bcbddbd4acc27ea5c51))
* Only use one clinical context of the bag ([`82a71df`](https://github.com/siemdejong/dpat/commit/82a71dfd2f25dfa12bebd26755c67e44c8feef75))

## v4.9.3 (2023-04-13)
### Fix
* Restructure config files ([`35c24a9`](https://github.com/siemdejong/dpat/commit/35c24a965cba5b34c920727bc5f6149f46a32c9a))
* Revert switch to cpu ([`0bb86e7`](https://github.com/siemdejong/dpat/commit/0bb86e783f67ad25765aa07aabdbd80a38f2f92d))

## v4.9.2 (2023-04-12)
### Fix
* Switch to clinical tiny BERT ([`ac199c1`](https://github.com/siemdejong/dpat/commit/ac199c152c25761eae3f1755e3c5b599cdde4502))

## v4.9.1 (2023-04-12)
### Fix
* Set llm to eval mode and disable gradient cal ([`1303353`](https://github.com/siemdejong/dpat/commit/1303353b0a59e9e6de9d3d1dc6ff9766c58ef082))

### Documentation
* Remove _static rom html_static_path ([`d1a1da8`](https://github.com/siemdejong/dpat/commit/d1a1da8fe8bdf79e0fe48e6426e004b45325f80c))

## v4.9.0 (2023-04-12)
### Feature
* Add location to splits output ([`4440a53`](https://github.com/siemdejong/dpat/commit/4440a53921d2182f0eed0a7de5ffd0a324795a44))
* Add CCMIL ([`b0294ce`](https://github.com/siemdejong/dpat/commit/b0294ceb96a52757eebf53566f72b527e9978294))
* Add clinical context to h5 dataset ([`3197f81`](https://github.com/siemdejong/dpat/commit/3197f8126dc74ba052a005c44fc4bc3b5e661a50))
* Add clinical context to image dataset ([`e229d1e`](https://github.com/siemdejong/dpat/commit/e229d1ef89d7216f406c94445f3ea7ea73df0e55))

### Fix
* Add typehints ([`486b392`](https://github.com/siemdejong/dpat/commit/486b392c6e4005e80c683a1d9bc057ee6ef0bd5e))
* Add LLM output last_hidden_state protocol ([`81e5e74`](https://github.com/siemdejong/dpat/commit/81e5e7445957409c1349b64c05cd2be1287b4de5))
* Ignore transformers missing import ([`5c6806d`](https://github.com/siemdejong/dpat/commit/5c6806d45f620d92db5971409410209dd426c49e))
* Add transformers dependency ([`8145723`](https://github.com/siemdejong/dpat/commit/8145723ea89f15199985be59a14f316561b1866b))

## v4.8.0 (2023-04-06)
### Feature
* Add num_workers to feature compiling ([`0f4ef30`](https://github.com/siemdejong/dpat/commit/0f4ef30378edc9397b74934674f6debd40552078))

## v4.7.0 (2023-04-06)
### Feature
* Attention tiles viz ([`df9f90f`](https://github.com/siemdejong/dpat/commit/df9f90fc734fddc4a6f269d02ca63b1a7422ba55))
* Change hparam search vars ([`fbd5847`](https://github.com/siemdejong/dpat/commit/fbd584767cee5a0d0bd6f22093df8b85e22db2d0))

### Fix
* Explicitly export attention model ([`7aa356d`](https://github.com/siemdejong/dpat/commit/7aa356d98842a681aba398ccecc1ee7e47332b81))
* Remove argument linking ([`d0d28e6`](https://github.com/siemdejong/dpat/commit/d0d28e6a3a04d9f76d155a226c6b1ff91d8380e1))
* Change defaults ([`dcdfea6`](https://github.com/siemdejong/dpat/commit/dcdfea6212b251c18bfbe5d4c8bbe06259c2a3ce))

## v4.6.0 (2023-04-04)
### Feature
* Make trials reproducible ([`6fb64c4`](https://github.com/siemdejong/dpat/commit/6fb64c493dba130ab64e4b44ece9a85d62846dd8))

## v4.5.0 (2023-04-04)
### Feature
* Test model ([`857d2cd`](https://github.com/siemdejong/dpat/commit/857d2cd7e6b11269a73c83a10625250f4229f61d))

## v4.4.0 (2023-03-30)
### Feature
* Add hparam tuner ([#31](https://github.com/siemdejong/dpat/issues/31)) ([`b2e4094`](https://github.com/siemdejong/dpat/commit/b2e40942abc3dad0b72499480fe024f56872b6ee))

### Fix
* Run black ([`619679f`](https://github.com/siemdejong/dpat/commit/619679fa3c808b699b16d79bd915d00bbcabca6b))

## v4.3.2 (2023-03-24)
### Fix
* Type error ([`e2af84a`](https://github.com/siemdejong/dpat/commit/e2af84a6474303aec2e2b8d25fdf47468a9cfb3b))
* Bug with reading from bytes ([`ba1c58e`](https://github.com/siemdejong/dpat/commit/ba1c58e5290c985877f99526936a7465c9244f4b))
* Give units to fraction logs ([`416749a`](https://github.com/siemdejong/dpat/commit/416749aa03e1c6d9fbee0ca350e9154033d0fc0e))

### Documentation
* Fix indent ([`2e7ebf1`](https://github.com/siemdejong/dpat/commit/2e7ebf1c03ac3fa04962cc694e6716070b31f0cc))
* Ignore autosummary ([`e63a009`](https://github.com/siemdejong/dpat/commit/e63a0094dd431241684169aebd858c1a3b92e350))
* Move docs from readme to gh pages ([`6012ebc`](https://github.com/siemdejong/dpat/commit/6012ebcadbe1315fcb0e010393290f6c1dc99215))
* Add docs to github pages ([`1f85ef8`](https://github.com/siemdejong/dpat/commit/1f85ef882464c87d89ee7109cb58ede34dccad5d))

## v4.3.1 (2023-03-23)
### Fix
* Up default patience to 100 ([`289d7f9`](https://github.com/siemdejong/dpat/commit/289d7f9b3b97702836f34a6f4815324cbad09247))

## v4.3.0 (2023-03-23)
### Feature
* Add early stopping ([`7c4eb01`](https://github.com/siemdejong/dpat/commit/7c4eb01e240402c6033fa2b06d802e41b3e90c60))
* Add dropout_p parameter to varmil ([`6b59e50`](https://github.com/siemdejong/dpat/commit/6b59e50d8b6c5c800ce0bdcadc09025159412dce))

## v4.2.0 (2023-03-23)
### Feature
* Add hidden features and dropout ([`452082c`](https://github.com/siemdejong/dpat/commit/452082cbff5e2f584938b7de84b887c0a035d418))

### Fix
* Typing ([`5dce7a4`](https://github.com/siemdejong/dpat/commit/5dce7a4bd174fa9e4411e506d11962b1eb495850))
* Add area under precision recall curve ([`1a2648f`](https://github.com/siemdejong/dpat/commit/1a2648faed30be99050f9d4877530070dffb238b))
* Remove optimizer/scheduler defaults ([`d2b62e1`](https://github.com/siemdejong/dpat/commit/d2b62e1631c88db8914497a55baf8a45d04a1f3a))
* Link max_epochs to scheduler T_max ([`1a05932`](https://github.com/siemdejong/dpat/commit/1a059322768140f33668e4481940bc9a94a6d582))
* Run black ([`db24156`](https://github.com/siemdejong/dpat/commit/db2415632a81eefd91db152848172511e381a4e1))
* Support lightning 2 ([`b4a822c`](https://github.com/siemdejong/dpat/commit/b4a822c166edf2ac2ff5d5557f3a8e09b8f68588))

## v4.1.0 (2023-03-23)
### Feature
* Entropymasker experiment ([`ec70eef`](https://github.com/siemdejong/dpat/commit/ec70eef917c2128d7e34ef50188b422bc262b732))

## v4.0.1 (2023-03-20)
### Fix
* Compiling model doesn't work, rollback ([`e733a24`](https://github.com/siemdejong/dpat/commit/e733a243879b26ebcf87b35a28d56b098db705b4))

## v4.0.0 (2023-03-20)
### Fix
* Update mil config to reflect new signature ([`c4dc0ed`](https://github.com/siemdejong/dpat/commit/c4dc0edce0acc9ad2453f874cde3b619e68ca559))
* Filter h5 dataset by filenames ([`b3e1770`](https://github.com/siemdejong/dpat/commit/b3e177062d58840a69271f323f84fa67d9bf31cc))
* Safely bypass sigint or sigterm ([`388d368`](https://github.com/siemdejong/dpat/commit/388d36824e27192a104b84f2a2629fea471b733e))
* Remove trainer arguments for pl2 ([`99194e0`](https://github.com/siemdejong/dpat/commit/99194e09417bd38a1f2c52d61b870199f91ab7db))

### Breaking
* train|val|test_path keywords now do not refer to the the target train|val|test paths to output h5 files to, but to the files providing paths_and_targets. file_path is the new target file, containing all images. ([`b3e1770`](https://github.com/siemdejong/dpat/commit/b3e177062d58840a69271f323f84fa67d9bf31cc))

## v3.2.1 (2023-03-17)
### Fix
* Remove pl2 removed trainer arguments ([`817beb8`](https://github.com/siemdejong/dpat/commit/817beb85c59ec8ae7bf3f0273f640c7086a12482))

## v3.2.0 (2023-03-17)
### Feature
* Automate masking with dlup entropy_masker ([`1066ea1`](https://github.com/siemdejong/dpat/commit/1066ea1264a7c56760b1fb4be10a7fced92336b6))
* Compile the pytorch lightning models ([`c009004`](https://github.com/siemdejong/dpat/commit/c009004acb837a3026cb370b927b461996f8f51c))

### Fix
* Lightning 2.0 ignore missing type imports ([`08b3e85`](https://github.com/siemdejong/dpat/commit/08b3e85608f0cc3171812f1cd1f5fe6f5e9a3bfa))
* Ddp default find_unused_parameters=False ([`3e2a15d`](https://github.com/siemdejong/dpat/commit/3e2a15dab7fd127b8f70c1439b2833a20b32ec4a))
* Use recommended precision ([`6b654e2`](https://github.com/siemdejong/dpat/commit/6b654e2fc26c7c1223529d1a42cd177ea8c95ff8))
* Rename *_epoch_end to on_*_epoch_end ([`37e0c52`](https://github.com/siemdejong/dpat/commit/37e0c52dc200bd72093c89aabc6016ec44cf7d60))
* Add torch/lightning v2 deps ([`2afdf05`](https://github.com/siemdejong/dpat/commit/2afdf057726f40caae1cae1f7fae99aa464ae415))
* Simclr-16-3 update ([`29fdf58`](https://github.com/siemdejong/dpat/commit/29fdf582a8c24a5475a605130c1903e1a5e30520))

### Performance
* Do not log to progress bar ([`da4e3b4`](https://github.com/siemdejong/dpat/commit/da4e3b487f24d1503fea17feb8280d486a5bdc36))

## v3.1.1 (2023-03-15)
### Fix
* Pretrained config typo ([`e6a3778`](https://github.com/siemdejong/dpat/commit/e6a37785d9b9ef49bdbf40eb009d83f35edff82e))

## v3.1.0 (2023-03-15)
### Feature
* Add num_subfolds parameter to create_splits ([`dfe7ceb`](https://github.com/siemdejong/dpat/commit/dfe7ceb3380fef3ea595cd3f7e68ed426b33989d))

### Fix
* Splits tests with num_subfolds ([`3a9d44e`](https://github.com/siemdejong/dpat/commit/3a9d44e84ae570e35fc2262a46fe4b467fc97f98))

## v3.0.1 (2023-03-15)
### Fix
* Update viz of untrained model ([`e32d9f5`](https://github.com/siemdejong/dpat/commit/e32d9f5a1bcacf45b3893c94ec3dddf6cd3682c2))

## v3.0.0 (2023-03-15)
### Feature
* From torchvision.models to pytorchcv models ([`bb5732d`](https://github.com/siemdejong/dpat/commit/bb5732d3f8b9fc0434720077d37a448df80e8670))
* Show targets/img_id/case_id distribution ([`c757995`](https://github.com/siemdejong/dpat/commit/c7579958dcb63b2664dc120ba1feac16af208489))

### Fix
* Pytorchcv not typed ignore missing imports ([`2b33f70`](https://github.com/siemdejong/dpat/commit/2b33f705443d4d2e92eb045f42f63efdba5feb34))

### Breaking
* in the config files, models must be specified by corresponding pytorchcv models, not torchvision. ([`bb5732d`](https://github.com/siemdejong/dpat/commit/bb5732d3f8b9fc0434720077d37a448df80e8670))

## v2.7.3 (2023-03-14)
### Performance
* Speed up feature compilation ([`772a0bd`](https://github.com/siemdejong/dpat/commit/772a0bd38e356b9d6b547e757286c0b973a45c2e))

## v2.7.2 (2023-03-14)
### Fix
* Create feature directory if not exists ([`674c8c4`](https://github.com/siemdejong/dpat/commit/674c8c4eb33ac614d03cee35a0993bec5b50b4e7))

## v2.7.1 (2023-03-13)
### Fix
* Bypass oom error ([`0177032`](https://github.com/siemdejong/dpat/commit/01770320bcae9608d4efc15793a9d53135e764f3))
* Add ipywidgets dependency ([`3c26dcd`](https://github.com/siemdejong/dpat/commit/3c26dcd0b43e868ff34bf424f0c1bf5c73bdc433))

## v2.7.0 (2023-03-10)
### Feature
* Visualize embedding distribution using t-sne ([`91c4190`](https://github.com/siemdejong/dpat/commit/91c4190f4477e2cb53fa5e855d1bf64cfd5a6ecf))

### Documentation
* Change installation order. ([`63d51c8`](https://github.com/siemdejong/dpat/commit/63d51c8b27db4711b141b9c4aa5104bc658aca57))

## v2.6.1 (2023-03-10)
### Fix
* Show case distribution ([`e2dcd66`](https://github.com/siemdejong/dpat/commit/e2dcd667eb1f48d00eb4fb2ce5ba653ec7149ebb))

## v2.6.0 (2023-03-10)
### Feature
* Nearest neighbours visualization features ([`6239607`](https://github.com/siemdejong/dpat/commit/6239607b792d561a4ca3ce5c424e3ac6b273b1ea))

## v2.5.2 (2023-03-10)
### Fix
* Jsonargparse dependency ([`94242b2`](https://github.com/siemdejong/dpat/commit/94242b2b7654ef4fb015f332a9c256a59c6d5aad))

## v2.5.1 (2023-03-10)
### Fix
* **log:** Log origin of image data ([`1bb50ae`](https://github.com/siemdejong/dpat/commit/1bb50aea5c76849fe9be7b5219abc6f938de9922))

## v2.5.0 (2023-03-09)
### Feature
* **pretrain:** Add SimCLR ([`502bba3`](https://github.com/siemdejong/dpat/commit/502bba37010819a4f0a90292e3e47e721aac8738))

## v2.4.2 (2023-03-08)
### Fix
* **type:** Fix __iter__ type error ([`d69fb24`](https://github.com/siemdejong/dpat/commit/d69fb24dee31c13cb3cd543b66526a9bdf4d87e9))
* Use WeightedRandomSampler ([`55334c4`](https://github.com/siemdejong/dpat/commit/55334c4ce402de9d39f49eea0ab39fbf99f98478))

## v2.4.1 (2023-03-08)
### Performance
* **varmil:** Set gradients to none instead of 0 ([`b8a850a`](https://github.com/siemdejong/dpat/commit/b8a850a9ce5ae8fe896585ff8eeb3d308049feb7))

## v2.4.0 (2023-03-07)
### Feature
* Train varmil ([`56a3ae9`](https://github.com/siemdejong/dpat/commit/56a3ae9ae1bddf481867ac146322ea9852e1cb2e))

### Fix
* Some typing issues ([`6a46853`](https://github.com/siemdejong/dpat/commit/6a46853ea016ee8dc8849b4dfb992a7d52cdedff))
* Bug where M was calculated wrong ([`d3a675b`](https://github.com/siemdejong/dpat/commit/d3a675b7befec145976f9a95774e591ef61ddd9e))
* Export variables from packages ([`e0db9e3`](https://github.com/siemdejong/dpat/commit/e0db9e3191c2455085ce2671694634cddc3cfde7))

## v2.3.3 (2023-03-04)
### Fix
* Uncomment feature compilation ([`4e018fe`](https://github.com/siemdejong/dpat/commit/4e018fe2c0760a1292edae65be11effd9c1f0787))

## v2.3.2 (2023-03-04)
### Fix
* Fix number of compiled feature vectors ([`b659501`](https://github.com/siemdejong/dpat/commit/b659501ccec8e7521e8c4d61d6e6e200a8956f40))

## v2.3.1 (2023-03-03)
### Fix
* Concatenate features ([`349b13d`](https://github.com/siemdejong/dpat/commit/349b13d5c0be43dea0ede14cffc65c11089c9154))
* H5 classmethod return type ([`bc4b24a`](https://github.com/siemdejong/dpat/commit/bc4b24af8dcd40caca4089217a5dd9c98ee2aae8))

## v2.3.0 (2023-03-02)
### Feature
* Add h5 dataset ([`0f6cfe2`](https://github.com/siemdejong/dpat/commit/0f6cfe24673dd27c43bd230636d4ad4698813865))

## v2.2.0 (2023-03-02)
### Feature
* Add feature extraction ([`e71d5be`](https://github.com/siemdejong/dpat/commit/e71d5becd0b1e194c5205d9993813dd74f4bb32d))

## v2.1.6 (2023-02-26)
### Performance
* Pin memory of dataloaders ([`4afcdf2`](https://github.com/siemdejong/dpat/commit/4afcdf2e59d0dd04aafbe90dfe070a831522d9ad))
* Zero_grad to non ([`d6d62c7`](https://github.com/siemdejong/dpat/commit/d6d62c759296b478c1038d94c7e103461a86ec5b))

## v2.1.5 (2023-02-26)
### Fix
* **tests:** Add tests for create_splits ([`780ed07`](https://github.com/siemdejong/dpat/commit/780ed077000f55c0f7b86888c1f9646ec0085864))

## v2.1.4 (2023-02-26)
### Fix
* **tests:** Add convert tests ([`819e23d`](https://github.com/siemdejong/dpat/commit/819e23dbdb36824ffbcf5a520e8d43ac42acb0d8))

## v2.1.3 (2023-02-25)
### Fix
* **semantic-release:** Add build command ([`c35a40f`](https://github.com/siemdejong/dpat/commit/c35a40f214390c36a4f377523fc03560f3eb8cfd))

## v2.1.2 (2023-02-25)
### Fix
* **typing:** Add typing ([`0842497`](https://github.com/siemdejong/dpat/commit/0842497f3498bf8f66af5ea917b5ecf012406b51))

## v2.1.1 (2023-02-23)
### Fix
* **swav:** Make swav return a number ([`d21dcce`](https://github.com/siemdejong/dpat/commit/d21dcce5c0590d94c87478b7c7eb8324476328f4))
* **deps:** Add tensorboard to dependencies ([`5ee3528`](https://github.com/siemdejong/dpat/commit/5ee35280bf5d7e990bcf3c26ae13975e5990b782))
* **convert:** Only log skip if skip_count>0 ([`0a58791`](https://github.com/siemdejong/dpat/commit/0a587910b9f2fa7b32d05c21aa8b0c3facb671b7))

## v2.1.0 (2023-02-22)
### Feature
* **data:** Add mean and std calculator ([`e6549f6`](https://github.com/siemdejong/dpat/commit/e6549f681bd0f4500077dd2365afd5a9a231c972))
* **stage1:** Implement swav with pl and lightly ([`3dcbdca`](https://github.com/siemdejong/dpat/commit/3dcbdcab5c4de8ebd08addb8c0f2ce02fdfa4803))

### Fix
* **package:** Add dependencies ([`b3d340c`](https://github.com/siemdejong/dpat/commit/b3d340c046929d3b06f417dc2ae40e61ac173e13))
* **package:** Package dpat subfolder ([`2fb78ed`](https://github.com/siemdejong/dpat/commit/2fb78ed751908f28a0ca1719018c8a74d114b956))

## v2.0.0 (2023-02-16)
### Feature
* **installation:** Remove the need for config.yml ([`bdbed64`](https://github.com/siemdejong/dpat/commit/bdbed641081d0717d97876aa3a76bb0a9f0c216f))

### Fix
* **dpat:** Only read config.yml if windows ([`f823ccc`](https://github.com/siemdejong/dpat/commit/f823ccca59c966622464f821da158a70e3449469))
* **logging:** Let the cli configure logging ([`85875fd`](https://github.com/siemdejong/dpat/commit/85875fd6684933dfc01431be8f007c45b266e0ca))

### Breaking
* installation via the config.yml is no longer possible. It is also no longer needed for splits/convert cli operations. For coming deep learning cli applications, it will be needed to fetch the path to vipsbin from a config with deep learning options. ([`bdbed64`](https://github.com/siemdejong/dpat/commit/bdbed641081d0717d97876aa3a76bb0a9f0c216f))
* logging with the config file is now unsupported. Configure logging in the application using the library. ([`85875fd`](https://github.com/siemdejong/dpat/commit/85875fd6684933dfc01431be8f007c45b266e0ca))

### Documentation
* **cuda:** Add docs about cuda ([`58ca68f`](https://github.com/siemdejong/dpat/commit/58ca68f97c7c6a956654a3f5cb2a6b694a5c2fdc))
* **logging:** Remove config.yml logging ([`60e6d90`](https://github.com/siemdejong/dpat/commit/60e6d906e27db69203db389d268ca6e846973db1))
* **readme:** Clarify log/vips config ([`0e06f66`](https://github.com/siemdejong/dpat/commit/0e06f66875dd5aa3344106d062aeb935f1ab1f02))

## v1.4.0 (2023-02-15)
### Feature
* **logging:** Add logging and log config ([`6c90359`](https://github.com/siemdejong/dpat/commit/6c90359a5d9153555a1c12a9277d6869ce113525))

### Fix
* **cli:** Remove unnecessary __name__=__main__ chk ([`c5f5223`](https://github.com/siemdejong/dpat/commit/c5f5223c4b1a6a5f8432f0406665a5fca35104d6))

### Documentation
* **convert:** Change bulk to batch ([`2272355`](https://github.com/siemdejong/dpat/commit/2272355593294d6b8963616dc02f295b44589be3))

## v1.3.0 (2023-02-15)
### Feature
* **dataset:** Add PMCHHGImageDataset ([`1c65f85`](https://github.com/siemdejong/dpat/commit/1c65f85f4e170171e27c6a104d2a82c9f0771149))

## v1.2.2 (2023-02-15)
### Fix
* **splits:** Change saved filename to relative ([`5ba6ed9`](https://github.com/siemdejong/dpat/commit/5ba6ed95ce33f92d4a9677af37c1645944b75edb))

## v1.2.1 (2023-02-15)
### Fix
* **convert:** Decompressionbomberror help ([`9ff923d`](https://github.com/siemdejong/dpat/commit/9ff923db4ec297c78daac5e3c8fb294e42072983))

### Documentation
* **cli:** Update docs of cli ([`b917e52`](https://github.com/siemdejong/dpat/commit/b917e52856c06a0dbbb094a53d591a971215e150))

## v1.2.0 (2023-02-15)
### Feature
* **cli:** Change from argeparse to click ([`470ae65`](https://github.com/siemdejong/dpat/commit/470ae65c54d90bc69cf93f764b3445c94433254d))

## v1.1.1 (2023-02-14)
### Fix
* **splits:** A bug with default include/exclude ([`3f46207`](https://github.com/siemdejong/dpat/commit/3f462075fa6a07adef252454a9069a478e7414d3))

## v1.1.0 (2023-02-14)
### Feature
* **splits:** Change split logic ([`5f38156`](https://github.com/siemdejong/dpat/commit/5f38156574a27694cd8d85f00f356bd335b50764))
* **splits:** Add inclusion and exclusion pattern ([`caa5d2a`](https://github.com/siemdejong/dpat/commit/caa5d2a3d17f01c738dcb6829f0f233cd0cc68cd))
* **splits:** Add overwrite argument ([`f2a664f`](https://github.com/siemdejong/dpat/commit/f2a664fdeb3c36495570aa452ca74a378861e9f1))

### Documentation
* **readme:** Change installation heading ([`dd1416d`](https://github.com/siemdejong/dpat/commit/dd1416df319638a642d1d28884c1fd34c81c9bb1))

## v1.0.0 (2023-02-14)
### Feature
* **dpat:** Change project.ini to config.yml ([`f13ed4b`](https://github.com/siemdejong/dpat/commit/f13ed4bcd9e44d7cbbfe74443effc8ff4347d38a))

### Breaking
* project.ini will not be read anymore. PATHS.vips has to be set in config.yml, somewhere near the root of the repository. ([`f13ed4b`](https://github.com/siemdejong/dpat/commit/f13ed4bcd9e44d7cbbfe74443effc8ff4347d38a))

### Documentation
* **installation:** Clarify where to specify vipsbin ([`643009a`](https://github.com/siemdejong/dpat/commit/643009ad8f01e10da84167df8792dc4baabcc160))
* **splits:** Add help of splits cli to README ([`e3ef258`](https://github.com/siemdejong/dpat/commit/e3ef25843576de0b9313bb58462d073a0b258a43))

## v0.4.1 (2023-02-14)
### Fix
* **dpat:** Import pyvips before openslide ([`2d25ae0`](https://github.com/siemdejong/dpat/commit/2d25ae00725775c880f7d3c568985f2a76b0c894))

## v0.4.0 (2023-02-14)
### Feature
* Add pre-commit ([`620dccb`](https://github.com/siemdejong/dpat/commit/620dccba985db8ab86ae6a6ea3fec3775b9aa057))
* **splits:** Create splits and run isort/black ([`fe8cf25`](https://github.com/siemdejong/dpat/commit/fe8cf25ac3940ea1810b66edea92bdf080eba792))

## v0.3.0 (2023-02-13)
### Feature
* Add num-workers and #chunks to cli ([`0f322c4`](https://github.com/siemdejong/dpat/commit/0f322c49bb3cbdaa7ecf961d85bb5c0c44f756f0))

### Documentation
* Fix typo ([`18971e2`](https://github.com/siemdejong/dpat/commit/18971e22813b39c2b3cb0679918e0f152f334e75))
* Fix 300fast typo ([`f0ed148`](https://github.com/siemdejong/dpat/commit/f0ed148fc62ad89d387fece03112bcf64c3cb383))
* Clarify image fn needs scanprogram for tif ([`f0f7384`](https://github.com/siemdejong/dpat/commit/f0f7384e8d548628b882099b2c3faebd380b7cd8))
* Clarify conda dependency ([`932569f`](https://github.com/siemdejong/dpat/commit/932569f78a459c218366a1d2ee3d6127a1a3323c))

## v0.2.0 (2023-02-13)
### Feature
* Clarify dlup will be installed with dpat ([`c062884`](https://github.com/siemdejong/dpat/commit/c062884fcd7232db34c8b48a3c72bd06be095589))

### Documentation
* Remove logo ([`f932b71`](https://github.com/siemdejong/dpat/commit/f932b714653002ea6b8023d1eb4f49032e90e3ae))
* Make package and add readme ([`e5fd4a5`](https://github.com/siemdejong/dpat/commit/e5fd4a5c44c160430e7245bc80b185300e6379bb))
