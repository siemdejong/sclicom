# CHANGELOG



## v5.1.0 (2024-03-05)

### Feature

* feat: allow building pages on demand ([`10259ca`](https://github.com/siemdejong/sclicom/commit/10259cadca2c4ba46b0bd2327ad16813f2421b10))

### Fix

* fix: make toml_version tuple ([`56e0f7a`](https://github.com/siemdejong/sclicom/commit/56e0f7af5d894702fdf32c08836b8ed70b30dcfe))

* fix: deactivate type check

This is to ignore new type errors. Issue is opened to fixe type errors ([`e9eb9a6`](https://github.com/siemdejong/sclicom/commit/e9eb9a6cac7e6994600c3d355e35d7467567c0de))

### Unknown

* Deactivate isort and black in ci ([`8d8cad6`](https://github.com/siemdejong/sclicom/commit/8d8cad60ee7d73e0a745a1dd006110725dbaa79b))

* Update README.md

Name change ([`b2f4eb3`](https://github.com/siemdejong/sclicom/commit/b2f4eb3928d579c7c770e8356e8fd70d1b5ca2db))


## v5.0.1 (2023-07-02)

### Fix

* fix: typing ([`e1e60f8`](https://github.com/siemdejong/sclicom/commit/e1e60f849fa8334bb91cdd382c286b5e83c9e74d))

* fix: typing ([`beaaf4d`](https://github.com/siemdejong/sclicom/commit/beaaf4d7c59126ade0ae79145cee8356e783d4ca))

### Unknown

* version at hand in ([`2205b58`](https://github.com/siemdejong/sclicom/commit/2205b58cdeb385c24b9e10c6d32aef7529dacfe0))


## v5.0.0 (2023-05-07)

### Breaking

* fix: remove scheduler arg

BREAKING CHANGE: scheduler is unset ([`03f60dc`](https://github.com/siemdejong/sclicom/commit/03f60dc22a90d7b60ecb3cb20ae8cb2624b4c4fb))

### Fix

* fix: do not track out, sbatch, ckpt ([`68d4c55`](https://github.com/siemdejong/sclicom/commit/68d4c557ba35da13bcdca11c9a46447292623a6f))

* fix: use train stage for setup ([`d0b63b1`](https://github.com/siemdejong/sclicom/commit/d0b63b1a7a2e72ffe5d1e8ab337924d049d6ec7c))

* fix: update dataloader ([`b166512`](https://github.com/siemdejong/sclicom/commit/b1665124193bf2c6832df7f9dcbb9c386f4e8c61))

* fix: update defaults ([`5964cc7`](https://github.com/siemdejong/sclicom/commit/5964cc758cd0a02cffcf13986bd67549c1a19622))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`6385a6a`](https://github.com/siemdejong/sclicom/commit/6385a6a55dfc26f0b33758a2979945e90eede4d6))


## v4.15.0 (2023-04-25)

### Feature

* feat: compute mean and confidence interval

Following TrivialAugment&#39;s recommendation to provide how the mean and
confidence interval are calculated. ([`da2e834`](https://github.com/siemdejong/sclicom/commit/da2e834abaf6bfbb745a356041277db9a34c9769))

### Fix

* fix: add confidence typehint ([`c057652`](https://github.com/siemdejong/sclicom/commit/c057652a1023cdfd954720bd84ac42d4635bb2a3))

* fix: add typehints ([`cb761f2`](https://github.com/siemdejong/sclicom/commit/cb761f2d610ede18c6afeeb13db1c30309e1f642))

* fix: revert hidden layers to one hidden layer

with dropout. ([`3139c46`](https://github.com/siemdejong/sclicom/commit/3139c46e737138cdae6e375d4432758a68373d72))

* fix: add typehint ([`f382d1f`](https://github.com/siemdejong/sclicom/commit/f382d1f7506a403304ff8260109e9b405a3d11e3))

* fix: ignore missing scipy imports ([`f562790`](https://github.com/siemdejong/sclicom/commit/f5627908133aef9d4997988d48a8ef12186d5b88))


## v4.14.2 (2023-04-25)

### Fix

* fix: remove contrastive collate_fn

fixes #13 ([`1811bb5`](https://github.com/siemdejong/sclicom/commit/1811bb50cc63742e63563f6aa2f94259abfc0a9b))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`d2acb15`](https://github.com/siemdejong/sclicom/commit/d2acb157ba1dd25c62eeb84a76ea517fdb9ddcc3))


## v4.14.1 (2023-04-24)

### Fix

* fix: don&#39;t try to open augmented set if not exists ([`bf33e12`](https://github.com/siemdejong/sclicom/commit/bf33e12ca961276f765c6f092f242c5bd919154c))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`1eafdef`](https://github.com/siemdejong/sclicom/commit/1eafdefab2aab062a7f54348b6f04c239d747684))


## v4.14.0 (2023-04-24)

### Feature

* feat: do data augmentation in feature space

Following extrapolation from https://arxiv.org/pdf/1702.05538.pdf. ([`2d0fa05`](https://github.com/siemdejong/sclicom/commit/2d0fa058f510ddc0eade62446d9f0a52803c717d))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`6fe7e1e`](https://github.com/siemdejong/sclicom/commit/6fe7e1ef0c32455609a482e25264e446da2e584e))


## v4.13.2 (2023-04-21)

### Fix

* fix: use clinical_context attr for export ([`446bb2b`](https://github.com/siemdejong/sclicom/commit/446bb2b87cd12eea5f2dc3024229de13f9ced49d))


## v4.13.1 (2023-04-21)

### Fix

* fix: alter seed config space ([`e8a652e`](https://github.com/siemdejong/sclicom/commit/e8a652e31f250686da0c345131cd863148f0c34c))

* fix: use variable splits_dirname ([`f3824e4`](https://github.com/siemdejong/sclicom/commit/f3824e41f8086a4c3e8aca29bbd8ef0f2fca57aa))

* fix: automatically set_clinical context ([`764b77c`](https://github.com/siemdejong/sclicom/commit/764b77cda317e4544b1475e77b57ff6f5f3101c6))


## v4.13.0 (2023-04-21)

### Feature

* feat: add device to mean and std calc ([`4db5601`](https://github.com/siemdejong/sclicom/commit/4db5601044ad3b8e652b6861602cd2d064c22716))

### Fix

* fix: add batch size to compile features tool ([`7a13c09`](https://github.com/siemdejong/sclicom/commit/7a13c094af585f0abd3b0acd9c88b7cb10663639))

* fix: add num_workers to compile features tool ([`018861f`](https://github.com/siemdejong/sclicom/commit/018861fc8740b57cd74260793d3113243799af04))

* fix: add clinical context to compile features tool ([`8d89607`](https://github.com/siemdejong/sclicom/commit/8d896075a73bc95cc08b1655f3e8830708c9d130))

* fix: change location dtype to string in hdf5 ([`ef9e5cf`](https://github.com/siemdejong/sclicom/commit/ef9e5cf1bd712721b5d9f99ced9a2d5054a1a3d0))

* fix: use normalization based on masked tiles ([`6e5e639`](https://github.com/siemdejong/sclicom/commit/6e5e639b4511561fa14131fddab465cb302bfb94))

* fix: typo ([`cadd918`](https://github.com/siemdejong/sclicom/commit/cadd918758f3c78558216e02396fc7ea4eca8279))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`41a1621`](https://github.com/siemdejong/sclicom/commit/41a162128013caef80a70d507db32bca4fa6cf90))


## v4.12.0 (2023-04-21)

### Feature

* feat: calculate mean and stddev ([`4ddf6d1`](https://github.com/siemdejong/sclicom/commit/4ddf6d1315486526885dd22a755e759e79d011e0))

### Fix

* fix: raise error if no mask_root_dir is set

when loading from disk ([`2fd7157`](https://github.com/siemdejong/sclicom/commit/2fd71577137ea13949e724ecd32bbb570418eb68))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`08bf1c9`](https://github.com/siemdejong/sclicom/commit/08bf1c9a221a7b022e5e8c6d8c041b53206f3561))


## v4.11.1 (2023-04-21)

### Documentation

* docs: add steps to adding-new-data ([`49f10d5`](https://github.com/siemdejong/sclicom/commit/49f10d5d1b303a4e12d1df39808195cae16edcfb))

* docs: clarify tensorflow for gpu installation ([`2a8b90d`](https://github.com/siemdejong/sclicom/commit/2a8b90d7aed04695171cce1607bacaaad66a5965))

### Fix

* fix: allow null as datamodule model ([`77dff7e`](https://github.com/siemdejong/sclicom/commit/77dff7e88daeb37f83df6128a7cd40c5b7063498))

* fix: typo

wrong number of selected images was displayed ([`a25a251`](https://github.com/siemdejong/sclicom/commit/a25a2512d2dc742f67c220b6d13f2c2098c31e79))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`466b68c`](https://github.com/siemdejong/sclicom/commit/466b68ca4fd7f04c62c50581b5b6225d0b74530f))


## v4.11.0 (2023-04-20)

### Feature

* feat: add denoising tool ([`ee43414`](https://github.com/siemdejong/sclicom/commit/ee434144fdff5becc756f4186fedfa851a49e746))

### Fix

* fix: add mask creation to CLI and update docs ([`0dbdbc4`](https://github.com/siemdejong/sclicom/commit/0dbdbc49a89204cc33f114bfe87f338a1472ea95))

### Unknown

* Merge pull request #43 from siemdejong/denoise

feat: add denoising tool ([`fb62c1b`](https://github.com/siemdejong/sclicom/commit/fb62c1b04145826ee99be17e5355855e58dd4c23))

* Merge pull request #42 from siemdejong/mask

fix: add mask creation to CLI and update docs ([`6121500`](https://github.com/siemdejong/sclicom/commit/6121500fd0bfa25bdbf50ef5e793444ca2c0836d))


## v4.10.1 (2023-04-14)

### Fix

* fix: change defaults ([`7240d9e`](https://github.com/siemdejong/sclicom/commit/7240d9e93fa030cd79eb0c1534931e1ac2520f70))


## v4.10.0 (2023-04-13)

### Feature

* feat: allow for CCMIL hparam tuning ([`9e587f8`](https://github.com/siemdejong/sclicom/commit/9e587f8765a36bc5e68525c567436ba922eeda31))

### Fix

* fix: typing of example array ([`cb98f39`](https://github.com/siemdejong/sclicom/commit/cb98f397736e6860e3427cc0ad92bbca303b0387))

* fix: make random tensor as example ([`a8d21f0`](https://github.com/siemdejong/sclicom/commit/a8d21f0440708b20894a659864a582f82328e072))

* fix: move tokenized inputs to device ([`2b75731`](https://github.com/siemdejong/sclicom/commit/2b75731c7e4dc5ac8e93ab322c7a8f0d039da54e))

* fix: make new example input for lightning ([`a431ab2`](https://github.com/siemdejong/sclicom/commit/a431ab2f82f2977900612bcbddbd4acc27ea5c51))

* fix: only use one clinical context of the bag ([`82a71df`](https://github.com/siemdejong/sclicom/commit/82a71dfd2f25dfa12bebd26755c67e44c8feef75))


## v4.9.3 (2023-04-13)

### Fix

* fix: restructure config files

CCMIL is used by default.
If another model is needed, specify with &#34;--model path/to/varmil.yaml&#34;
when training the model. ([`35c24a9`](https://github.com/siemdejong/sclicom/commit/35c24a965cba5b34c920727bc5f6149f46a32c9a))

* fix: revert switch to cpu ([`0bb86e7`](https://github.com/siemdejong/sclicom/commit/0bb86e783f67ad25765aa07aabdbd80a38f2f92d))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`47663d2`](https://github.com/siemdejong/sclicom/commit/47663d278a2058468cb5e6119348690d816798b9))


## v4.9.2 (2023-04-12)

### Fix

* fix: switch to clinical tiny BERT

Because it is much quicker. ([`ac199c1`](https://github.com/siemdejong/sclicom/commit/ac199c152c25761eae3f1755e3c5b599cdde4502))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`a4d16c2`](https://github.com/siemdejong/sclicom/commit/a4d16c21f51141edd5ea12b4f34600c5afec11e4))


## v4.9.1 (2023-04-12)

### Ci

* ci: disable HDF5 version check

To bypass &#34;UserWarning: h5py is running against HDF5 1.14.0 when it was
built against 1.12.2&#34;. ([`148c8ab`](https://github.com/siemdejong/sclicom/commit/148c8ab37e8c5168dd0ed72df6dea0b44d9abb67))

### Documentation

* docs: remove _static rom html_static_path ([`d1a1da8`](https://github.com/siemdejong/sclicom/commit/d1a1da8fe8bdf79e0fe48e6426e004b45325f80c))

### Fix

* fix: set llm to eval mode and disable gradient cal

unless trainable_llm is set to True. ([`1303353`](https://github.com/siemdejong/sclicom/commit/1303353b0a59e9e6de9d3d1dc6ff9766c58ef082))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`cafd5bf`](https://github.com/siemdejong/sclicom/commit/cafd5bfbed0574a7be579c15a066b1189c58ee1b))


## v4.9.0 (2023-04-12)

### Feature

* feat: add location to splits output ([`4440a53`](https://github.com/siemdejong/sclicom/commit/4440a53921d2182f0eed0a7de5ffd0a324795a44))

* feat: add CCMIL ([`b0294ce`](https://github.com/siemdejong/sclicom/commit/b0294ceb96a52757eebf53566f72b527e9978294))

* feat: add clinical context to h5 dataset ([`3197f81`](https://github.com/siemdejong/sclicom/commit/3197f8126dc74ba052a005c44fc4bc3b5e661a50))

* feat: add clinical context to image dataset ([`e229d1e`](https://github.com/siemdejong/sclicom/commit/e229d1ef89d7216f406c94445f3ea7ea73df0e55))

### Fix

* fix: add typehints ([`486b392`](https://github.com/siemdejong/sclicom/commit/486b392c6e4005e80c683a1d9bc057ee6ef0bd5e))

* fix: add LLM output last_hidden_state protocol ([`81e5e74`](https://github.com/siemdejong/sclicom/commit/81e5e7445957409c1349b64c05cd2be1287b4de5))

* fix: ignore transformers missing import ([`5c6806d`](https://github.com/siemdejong/sclicom/commit/5c6806d45f620d92db5971409410209dd426c49e))

* fix: add transformers dependency ([`8145723`](https://github.com/siemdejong/sclicom/commit/8145723ea89f15199985be59a14f316561b1866b))

### Unknown

* tests: add location to splits ([`2a9698a`](https://github.com/siemdejong/sclicom/commit/2a9698aed3ffaf58ca648894aea6d68b93b8eafd))

* nb: add clinical context to existing hdf5 files ([`906b370`](https://github.com/siemdejong/sclicom/commit/906b37099d4b11beb72352113a0232f7bacab2e6))

* nb: test clinnical text embeddings ([`7a48701`](https://github.com/siemdejong/sclicom/commit/7a48701f9436afad0b9bd79a1e8aad6fb14d66af))

* Merge branch &#39;main&#39; of https://github.com/siemdejong/dpat ([`6e56e30`](https://github.com/siemdejong/sclicom/commit/6e56e30a95167122d605d80fc8f3e548c1cd87a4))


## v4.8.0 (2023-04-06)

### Feature

* feat: add num_workers to feature compiling ([`0f4ef30`](https://github.com/siemdejong/sclicom/commit/0f4ef30378edc9397b74934674f6debd40552078))


## v4.7.0 (2023-04-06)

### Feature

* feat: attention tiles viz ([`df9f90f`](https://github.com/siemdejong/sclicom/commit/df9f90fc734fddc4a6f269d02ca63b1a7422ba55))

* feat: change hparam search vars ([`fbd5847`](https://github.com/siemdejong/sclicom/commit/fbd584767cee5a0d0bd6f22093df8b85e22db2d0))

### Fix

* fix: explicitly export attention model ([`7aa356d`](https://github.com/siemdejong/sclicom/commit/7aa356d98842a681aba398ccecc1ee7e47332b81))

* fix: remove argument linking ([`d0d28e6`](https://github.com/siemdejong/sclicom/commit/d0d28e6a3a04d9f76d155a226c6b1ff91d8380e1))

* fix: change defaults ([`dcdfea6`](https://github.com/siemdejong/sclicom/commit/dcdfea6212b251c18bfbe5d4c8bbe06259c2a3ce))

### Unknown

* Update issue templates

gh: add docs request issue template ([`7d0663c`](https://github.com/siemdejong/sclicom/commit/7d0663cc4985be3273aa1cd6aaa413e545e1c875))


## v4.6.0 (2023-04-04)

### Feature

* feat: make trials reproducible

by adding a seed

and reconfigure search space ([`6fb64c4`](https://github.com/siemdejong/sclicom/commit/6fb64c493dba130ab64e4b44ece9a85d62846dd8))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`86bdd62`](https://github.com/siemdejong/sclicom/commit/86bdd62bb7c90f97a766741c135e19339b2c7775))


## v4.5.0 (2023-04-04)

### Feature

* feat: test model ([`857d2cd`](https://github.com/siemdejong/sclicom/commit/857d2cd7e6b11269a73c83a10625250f4229f61d))

### Unknown

* Merge pull request #35 from siemdejong/test

Test the model ([`cad0f37`](https://github.com/siemdejong/sclicom/commit/cad0f375e519438e7722615776c4e3dfd2a8633f))


## v4.4.0 (2023-03-30)

### Ci

* ci: only run sphinx build when release successful ([`e4d2c53`](https://github.com/siemdejong/sclicom/commit/e4d2c534549c5b7bbd8346f0a038c893a0b3148d))

### Feature

* feat: add hparam tuner (#31)

* feat: add hparam tuner

* docs: add clarification on wprkaround class

* fix: don&#39;t assume self.D is a list

* fix: move parameters to one place

* fix: remove ray_lightning

* fix: use minimum number of epochs

* fix: use multivariate TPE with constant_liar

* fix: report accuracies to ray

* fix: add comments on tpesampler ([`b2e4094`](https://github.com/siemdejong/sclicom/commit/b2e40942abc3dad0b72499480fe024f56872b6ee))

### Fix

* fix: run black ([`619679f`](https://github.com/siemdejong/sclicom/commit/619679fa3c808b699b16d79bd915d00bbcabca6b))

### Unknown

* Optuna (#33)

* feat: add hparam tuner

* docs: add clarification on wprkaround class

* fix: don&#39;t assume self.D is a list

* fix: move parameters to one place

* fix: remove ray_lightning

* fix: use minimum number of epochs

* fix: use multivariate TPE with constant_liar

* fix: report accuracies to ray

* fix: add comments on tpesampler

* fix: run precommit ([`53f16b4`](https://github.com/siemdejong/sclicom/commit/53f16b4576a0b4af7237232e6d123f97c1126a85))


## v4.3.2 (2023-03-24)

### Ci

* ci: only build docs if release workflow completed ([`9d49a34`](https://github.com/siemdejong/sclicom/commit/9d49a3413bc698ff4fcec3bc4e0e595e3f8a5b51))

* ci: remove sphinx action ([`b512b35`](https://github.com/siemdejong/sclicom/commit/b512b35851ec923262aba952ead1b54a142bdbf1))

* ci: set bash defaults ([`18bbd18`](https://github.com/siemdejong/sclicom/commit/18bbd182e323ed5c0b6c90d38bbc709399dfbc56))

* ci: activate dpat for docs compilation ([`f459363`](https://github.com/siemdejong/sclicom/commit/f45936357974170c7370e85f2e2176841f0b16e7))

### Documentation

* docs: fix indent ([`2e7ebf1`](https://github.com/siemdejong/sclicom/commit/2e7ebf1c03ac3fa04962cc694e6716070b31f0cc))

* docs: ignore autosummary ([`e63a009`](https://github.com/siemdejong/sclicom/commit/e63a0094dd431241684169aebd858c1a3b92e350))

* docs: move docs from readme to gh pages ([`6012ebc`](https://github.com/siemdejong/sclicom/commit/6012ebcadbe1315fcb0e010393290f6c1dc99215))

* docs: add docs to github pages ([`1f85ef8`](https://github.com/siemdejong/sclicom/commit/1f85ef882464c87d89ee7109cb58ede34dccad5d))

### Fix

* fix: type error ([`e2af84a`](https://github.com/siemdejong/sclicom/commit/e2af84a6474303aec2e2b8d25fdf47468a9cfb3b))

* fix: bug with reading from bytes

Seek first character of a string before reading it in the h5 dataset. ([`ba1c58e`](https://github.com/siemdejong/sclicom/commit/ba1c58e5290c985877f99526936a7465c9244f4b))

* fix: give units to fraction logs ([`416749a`](https://github.com/siemdejong/sclicom/commit/416749aa03e1c6d9fbee0ca350e9154033d0fc0e))


## v4.3.1 (2023-03-23)

### Fix

* fix: up default patience to 100 ([`289d7f9`](https://github.com/siemdejong/sclicom/commit/289d7f9b3b97702836f34a6f4815324cbad09247))


## v4.3.0 (2023-03-23)

### Feature

* feat: add early stopping ([`7c4eb01`](https://github.com/siemdejong/sclicom/commit/7c4eb01e240402c6033fa2b06d802e41b3e90c60))

* feat: add dropout_p parameter to varmil

To possibly change the amount of dropout ([`6b59e50`](https://github.com/siemdejong/sclicom/commit/6b59e50d8b6c5c800ce0bdcadc09025159412dce))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`73c584b`](https://github.com/siemdejong/sclicom/commit/73c584b2087de41c69dbb9654fbdccf5d85b68a7))


## v4.2.0 (2023-03-23)

### Ci

* ci: bypass conda bug

https://stackoverflow.com/a/72178361 ([`83c3991`](https://github.com/siemdejong/sclicom/commit/83c3991fbd94c5d0e767f9ea6b645fb98aa50af6))

### Feature

* feat: add hidden features and dropout ([`452082c`](https://github.com/siemdejong/sclicom/commit/452082cbff5e2f584938b7de84b887c0a035d418))

### Fix

* fix: typing ([`5dce7a4`](https://github.com/siemdejong/sclicom/commit/5dce7a4bd174fa9e4411e506d11962b1eb495850))

* fix: add area under precision recall curve ([`1a2648f`](https://github.com/siemdejong/sclicom/commit/1a2648faed30be99050f9d4877530070dffb238b))

* fix: remove optimizer/scheduler defaults

This was needed to set the optimizer and scheduler via the command line, like
&#34;--optimizer=Adam --optimizer.lr=0.003&#34; etc. ([`d2b62e1`](https://github.com/siemdejong/sclicom/commit/d2b62e1631c88db8914497a55baf8a45d04a1f3a))

* fix: link max_epochs to scheduler T_max

Because otherwise the cosine annealing doesn&#39;t work and only uses the
initial amount, regardless of what is set at the command line. ([`1a05932`](https://github.com/siemdejong/sclicom/commit/1a059322768140f33668e4481940bc9a94a6d582))

* fix: run black ([`db24156`](https://github.com/siemdejong/sclicom/commit/db2415632a81eefd91db152848172511e381a4e1))

* fix: support lightning 2 ([`b4a822c`](https://github.com/siemdejong/sclicom/commit/b4a822c166edf2ac2ff5d5557f3a8e09b8f68588))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`6c3e27c`](https://github.com/siemdejong/sclicom/commit/6c3e27c602c2ab3da8e55afe29c910c5ba1860fa))


## v4.1.0 (2023-03-23)

### Feature

* feat: entropymasker experiment ([`ec70eef`](https://github.com/siemdejong/sclicom/commit/ec70eef917c2128d7e34ef50188b422bc262b732))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`62e2f10`](https://github.com/siemdejong/sclicom/commit/62e2f10da3790d917616da06e0a88ab8971e10be))


## v4.0.1 (2023-03-20)

### Fix

* fix: compiling model doesn&#39;t work, rollback

ref: see snellius output slurm-2473416.out ([`e733a24`](https://github.com/siemdejong/sclicom/commit/e733a243879b26ebcf87b35a28d56b098db705b4))


## v4.0.0 (2023-03-20)

### Breaking

* fix: filter h5 dataset by filenames

Using the paths_and_targets keyword, which is equivalent to the similar
similar keyword in the tile dataset.

BREAKING CHANGE: train|val|test_path keywords now do not refer to the
the target train|val|test paths to output h5 files to, but to the files
providing paths_and_targets.
file_path is the new target file, containing all images. ([`b3e1770`](https://github.com/siemdejong/sclicom/commit/b3e177062d58840a69271f323f84fa67d9bf31cc))

### Fix

* fix: update mil config to reflect new signature ([`c4dc0ed`](https://github.com/siemdejong/sclicom/commit/c4dc0edce0acc9ad2453f874cde3b619e68ca559))

* fix: safely bypass sigint or sigterm

handy for using with slurm for example ([`388d368`](https://github.com/siemdejong/sclicom/commit/388d36824e27192a104b84f2a2629fea471b733e))

* fix: remove trainer arguments for pl2 ([`99194e0`](https://github.com/siemdejong/sclicom/commit/99194e09417bd38a1f2c52d61b870199f91ab7db))


## v3.2.1 (2023-03-17)

### Fix

* fix: remove pl2 removed trainer arguments ([`817beb8`](https://github.com/siemdejong/sclicom/commit/817beb85c59ec8ae7bf3f0273f640c7086a12482))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`708e3fb`](https://github.com/siemdejong/sclicom/commit/708e3fbb5a65ab05d63b211a6c4b5635490a6a1f))


## v3.2.0 (2023-03-17)

### Feature

* feat: automate masking with dlup entropy_masker ([`1066ea1`](https://github.com/siemdejong/sclicom/commit/1066ea1264a7c56760b1fb4be10a7fced92336b6))

* feat: compile the pytorch lightning models ([`c009004`](https://github.com/siemdejong/sclicom/commit/c009004acb837a3026cb370b927b461996f8f51c))

### Fix

* fix: lightning 2.0 ignore missing type imports ([`08b3e85`](https://github.com/siemdejong/sclicom/commit/08b3e85608f0cc3171812f1cd1f5fe6f5e9a3bfa))

* fix: ddp default find_unused_parameters=False

Lightning 2+ sets find_unused_parameters for the ddp strategy to false.
Therefore, it is unnecessary to set it to False in the config file. ([`3e2a15d`](https://github.com/siemdejong/sclicom/commit/3e2a15dab7fd127b8f70c1439b2833a20b32ec4a))

* fix: use recommended precision

bf16-mixed has less ambiguity than bf16. bf16-mixed makes it clear
mixed precision is used. ([`6b654e2`](https://github.com/siemdejong/sclicom/commit/6b654e2fc26c7c1223529d1a42cd177ea8c95ff8))

* fix: rename *_epoch_end to on_*_epoch_end

*_epoch_end is deprecated in lightning 2 ([`37e0c52`](https://github.com/siemdejong/sclicom/commit/37e0c52dc200bd72093c89aabc6016ec44cf7d60))

* fix: add torch/lightning v2 deps ([`2afdf05`](https://github.com/siemdejong/sclicom/commit/2afdf057726f40caae1cae1f7fae99aa464ae415))

* fix: simclr-16-3 update ([`29fdf58`](https://github.com/siemdejong/sclicom/commit/29fdf582a8c24a5475a605130c1903e1a5e30520))

### Performance

* perf: do not log to progress bar

Following the lightning 2.0 recommendation of not logging to
the progress bar, AUC and F1 are not logged to the process bar anymore.
They are still logged to the installed logger, e.g. Tensorboard. ([`da4e3b4`](https://github.com/siemdejong/sclicom/commit/da4e3b487f24d1503fea17feb8280d486a5bdc36))

### Unknown

* Mask load from disk (#25)

* feat: mask tiles in pmchhgImageDataset

Using the new entropy_masker mask function from dlup.

* fix: bumpy python version ([`340edde`](https://github.com/siemdejong/sclicom/commit/340eddec2dec6db48f6488caac6315f1400b5653))

* Merge pull request #24 from siemdejong/torchv2

Torchv2 ([`a0ef974`](https://github.com/siemdejong/sclicom/commit/a0ef97466a35a0fc853b2f4af864fbd308744fc6))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`51c0fad`](https://github.com/siemdejong/sclicom/commit/51c0fadb6794befcec87e5e1b34b676165e23aac))


## v3.1.1 (2023-03-15)

### Fix

* fix: pretrained config typo ([`e6a3778`](https://github.com/siemdejong/sclicom/commit/e6a37785d9b9ef49bdbf40eb009d83f35edff82e))


## v3.1.0 (2023-03-15)

### Feature

* feat: add num_subfolds parameter to create_splits ([`dfe7ceb`](https://github.com/siemdejong/sclicom/commit/dfe7ceb3380fef3ea595cd3f7e68ed426b33989d))

### Fix

* fix: splits tests with num_subfolds ([`3a9d44e`](https://github.com/siemdejong/sclicom/commit/3a9d44e84ae570e35fc2262a46fe4b467fc97f98))

### Unknown

* Merge pull request #23 from siemdejong/less-folds

feat: add num_subfolds parameter to create_splits

fix #22 ([`5d201d1`](https://github.com/siemdejong/sclicom/commit/5d201d1204b3fb77b8ff74314cc697d9f81914e9))


## v3.0.1 (2023-03-15)

### Fix

* fix: update viz of untrained model ([`e32d9f5`](https://github.com/siemdejong/sclicom/commit/e32d9f5a1bcacf45b3893c94ec3dddf6cd3682c2))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`d63095a`](https://github.com/siemdejong/sclicom/commit/d63095af26052f9a4e81944e401473a929584ebf))


## v3.0.0 (2023-03-15)

### Breaking

* feat: from torchvision.models to pytorchcv models

Torchvision has less pretrained and prebuilt models available than
pytorchcv.
Fort the application of a small dataset, it is nice to have smaller
resnets available, which torchvision has less support for.

BREAKING CHANGE: in the config files, models must be specified by
corresponding pytorchcv models, not torchvision. ([`bb5732d`](https://github.com/siemdejong/sclicom/commit/bb5732d3f8b9fc0434720077d37a448df80e8670))

### Feature

* feat: show targets/img_id/case_id distribution ([`c757995`](https://github.com/siemdejong/sclicom/commit/c7579958dcb63b2664dc120ba1feac16af208489))

### Fix

* fix: pytorchcv not typed ignore missing imports ([`2b33f70`](https://github.com/siemdejong/sclicom/commit/2b33f705443d4d2e92eb045f42f63efdba5feb34))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`79c5b72`](https://github.com/siemdejong/sclicom/commit/79c5b7211b9a4cc5933d2b2387730c8ef4483466))


## v2.7.3 (2023-03-14)

### Performance

* perf: speed up feature compilation

Fixes part of #16, namely the writing part. ([`772a0bd`](https://github.com/siemdejong/sclicom/commit/772a0bd38e356b9d6b547e757286c0b973a45c2e))


## v2.7.2 (2023-03-14)

### Fix

* fix: create feature directory if not exists ([`674c8c4`](https://github.com/siemdejong/sclicom/commit/674c8c4eb33ac614d03cee35a0993bec5b50b4e7))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`0859ba1`](https://github.com/siemdejong/sclicom/commit/0859ba12ca95f65c2481da080ce784892d6bb186))


## v2.7.1 (2023-03-13)

### Fix

* fix: bypass oom error ([`0177032`](https://github.com/siemdejong/sclicom/commit/01770320bcae9608d4efc15793a9d53135e764f3))

* fix: add ipywidgets dependency

for tqdm in notebooks ([`3c26dcd`](https://github.com/siemdejong/sclicom/commit/3c26dcd0b43e868ff34bf424f0c1bf5c73bdc433))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`65a9bd1`](https://github.com/siemdejong/sclicom/commit/65a9bd135a59d1c82535f4621e2c303bd26846be))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`07c8f20`](https://github.com/siemdejong/sclicom/commit/07c8f20146781ce3ceee18f54f5379057dd34d52))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`120fefb`](https://github.com/siemdejong/sclicom/commit/120fefbff711643c52f4242d46c332d8d87d1ad9))


## v2.7.0 (2023-03-10)

### Documentation

* docs: change installation order. ([`63d51c8`](https://github.com/siemdejong/sclicom/commit/63d51c8b27db4711b141b9c4aa5104bc658aca57))

### Feature

* feat: visualize embedding distribution using t-sne ([`91c4190`](https://github.com/siemdejong/sclicom/commit/91c4190f4477e2cb53fa5e855d1bf64cfd5a6ecf))


## v2.6.1 (2023-03-10)

### Chore

* chore: remove notebooks from linguist ([`98e5411`](https://github.com/siemdejong/sclicom/commit/98e5411bcc96250b04a792ee026056f3bc8e8fcd))

### Fix

* fix: show case distribution ([`e2dcd66`](https://github.com/siemdejong/sclicom/commit/e2dcd667eb1f48d00eb4fb2ce5ba653ec7149ebb))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`4c2edf3`](https://github.com/siemdejong/sclicom/commit/4c2edf300882335862dd1ba8c108690d39c312e5))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`28b5398`](https://github.com/siemdejong/sclicom/commit/28b5398feeb75a2709af212662b1b72308a0754c))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`90a8b81`](https://github.com/siemdejong/sclicom/commit/90a8b8131d4cb6ea277658b6531d999995a2774d))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`1b81024`](https://github.com/siemdejong/sclicom/commit/1b810244d81e648641e387e0e42510840d51b285))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`2639798`](https://github.com/siemdejong/sclicom/commit/26397983f1843b69489a5a80f3e13e3582b2a5a1))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`bdbc689`](https://github.com/siemdejong/sclicom/commit/bdbc6898fff46cb21bb52fa6b3bc8517ca3a1acc))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`77ef0eb`](https://github.com/siemdejong/sclicom/commit/77ef0ebe48d41698c9de90e5811735d34448c50b))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`0ca5433`](https://github.com/siemdejong/sclicom/commit/0ca5433aba601fd6f1109758b13c8fe890a46a44))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`f9152ee`](https://github.com/siemdejong/sclicom/commit/f9152ee0ffcb5aecb9ef06fa09761c4546abaa2b))

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`1cc30c4`](https://github.com/siemdejong/sclicom/commit/1cc30c4727626439e582b722d54d334d91ece9c0))


## v2.6.0 (2023-03-10)

### Feature

* feat: nearest neighbours visualization features ([`6239607`](https://github.com/siemdejong/sclicom/commit/6239607b792d561a4ca3ce5c424e3ac6b273b1ea))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`b47b685`](https://github.com/siemdejong/sclicom/commit/b47b685b08b14e4dd44b09e05067f7423e9a021d))


## v2.5.2 (2023-03-10)

### Fix

* fix: jsonargparse dependency ([`94242b2`](https://github.com/siemdejong/sclicom/commit/94242b2b7654ef4fb015f332a9c256a59c6d5aad))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`8440021`](https://github.com/siemdejong/sclicom/commit/844002110e4c8ffff34378183ca07610bcafb075))


## v2.5.1 (2023-03-10)

### Fix

* fix(log): log origin of image data ([`1bb50ae`](https://github.com/siemdejong/sclicom/commit/1bb50aea5c76849fe9be7b5219abc6f938de9922))


## v2.5.0 (2023-03-09)

### Feature

* feat(pretrain): add SimCLR ([`502bba3`](https://github.com/siemdejong/sclicom/commit/502bba37010819a4f0a90292e3e47e721aac8738))

### Unknown

* Merge pull request #21 from siemdejong/simclr

feat(pretrain): add SimCLR ([`c6dc62f`](https://github.com/siemdejong/sclicom/commit/c6dc62fd807dc639f6fea911ee15190d41625450))


## v2.4.2 (2023-03-08)

### Ci

* ci: run mypy in the dpat directory at pre-commit ([`662276e`](https://github.com/siemdejong/sclicom/commit/662276e9450f0c7af6d2818f1d2e3461d5f7bc10))

### Fix

* fix(type): fix __iter__ type error ([`d69fb24`](https://github.com/siemdejong/sclicom/commit/d69fb24dee31c13cb3cd543b66526a9bdf4d87e9))

* fix: use WeightedRandomSampler

Because the training dataset could be imbalanced. ([`55334c4`](https://github.com/siemdejong/sclicom/commit/55334c4ce402de9d39f49eea0ab39fbf99f98478))

### Unknown

* Merge pull request #20 from siemdejong/oversampling-minority

fix: use WeightedRandomSampler

Fixes #19 ([`f20f57e`](https://github.com/siemdejong/sclicom/commit/f20f57e5e1d152e323eb7bc1f93fab81eb72f764))


## v2.4.1 (2023-03-08)

### Performance

* perf(varmil): set gradients to none instead of 0 ([`b8a850a`](https://github.com/siemdejong/sclicom/commit/b8a850a9ce5ae8fe896585ff8eeb3d308049feb7))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`a344d8f`](https://github.com/siemdejong/sclicom/commit/a344d8fc82ee5c1a1c32ebd53993ee3a8a067a9c))


## v2.4.0 (2023-03-07)

### Ci

* ci: rm arch name from ci cache ([`488edfa`](https://github.com/siemdejong/sclicom/commit/488edfa54f1086d8108634fa864a70aeff5861d8))

### Feature

* feat: train varmil ([`56a3ae9`](https://github.com/siemdejong/sclicom/commit/56a3ae9ae1bddf481867ac146322ea9852e1cb2e))

### Fix

* fix: some typing issues ([`6a46853`](https://github.com/siemdejong/sclicom/commit/6a46853ea016ee8dc8849b4dfb992a7d52cdedff))

* fix: bug where M was calculated wrong ([`d3a675b`](https://github.com/siemdejong/sclicom/commit/d3a675b7befec145976f9a95774e591ef61ddd9e))

* fix: export variables from packages ([`e0db9e3`](https://github.com/siemdejong/sclicom/commit/e0db9e3191c2455085ce2671694634cddc3cfde7))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`fcaa977`](https://github.com/siemdejong/sclicom/commit/fcaa977a9ddaa6a17637aa0e679fac0ddb56de36))


## v2.3.3 (2023-03-04)

### Ci

* ci: add env caching ([`c8bef7e`](https://github.com/siemdejong/sclicom/commit/c8bef7eb8358de4603bc849d418e7f556cab9f36))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`fca3841`](https://github.com/siemdejong/sclicom/commit/fca3841293aa381c545c752ab4857fea7bc1fb46))


## v2.3.2 (2023-03-04)

### Ci

* ci: flake8 for extract_features pkg ([`c84b6da`](https://github.com/siemdejong/sclicom/commit/c84b6daebedfd17ccd867d0e286b6b041c735fcd))

### Fix

* fix: uncomment feature compilation ([`4e018fe`](https://github.com/siemdejong/sclicom/commit/4e018fe2c0760a1292edae65be11effd9c1f0787))

* fix: fix number of compiled feature vectors

Fix #17 ([`b659501`](https://github.com/siemdejong/sclicom/commit/b659501ccec8e7521e8c4d61d6e6e200a8956f40))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`dbb61cf`](https://github.com/siemdejong/sclicom/commit/dbb61cf71d8b716b1853aec18ac51d7ca12198bd))


## v2.3.1 (2023-03-03)

### Fix

* fix: concatenate features

Feature vectors of tiles belonging to the same image will be concatenated and
stored in the belonging image group along with their metadata in datasets
instead of h5-attrs.

This allows for MIL to use batch size=1bag
and load one dataset at a time. ([`349b13d`](https://github.com/siemdejong/sclicom/commit/349b13d5c0be43dea0ede14cffc65c11089c9154))

* fix: h5 classmethod return type ([`bc4b24a`](https://github.com/siemdejong/sclicom/commit/bc4b24af8dcd40caca4089217a5dd9c98ee2aae8))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`05dc1e9`](https://github.com/siemdejong/sclicom/commit/05dc1e92d720737e5b98da8e2106ed07f3126637))


## v2.3.0 (2023-03-02)

### Feature

* feat: add h5 dataset

Multi instance learning in stage 2 is easier on tensors obtained from an
hdf5 file with datasets embedded. ([`0f6cfe2`](https://github.com/siemdejong/sclicom/commit/0f6cfe24673dd27c43bd230636d4ad4698813865))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`87ce747`](https://github.com/siemdejong/sclicom/commit/87ce747197f8229f907f16668137e68595b8995f))


## v2.2.0 (2023-03-02)

### Feature

* feat: add feature extraction

Add feature extractor training.
Add feature vector compilation to HDF5.
Add config files to be used with pl.Trainer.
Refactor.
Add float32 matmul precision setting.
Add cudnn auto tuner setting.
Add Omegaconf &#34;now&#34; interpolation.
Require lightning[extra] and h5py as deps.
Run pre-commit also on dpat/extract_features.
Link pl CLI to pl Trainers. ([`e71d5be`](https://github.com/siemdejong/sclicom/commit/e71d5becd0b1e194c5205d9993813dd74f4bb32d))


## v2.1.6 (2023-02-26)

### Performance

* perf: pin memory of dataloaders ([`4afcdf2`](https://github.com/siemdejong/sclicom/commit/4afcdf2e59d0dd04aafbe90dfe070a831522d9ad))

* perf: zero_grad to non ([`d6d62c7`](https://github.com/siemdejong/sclicom/commit/d6d62c759296b478c1038d94c7e103461a86ec5b))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`df2974e`](https://github.com/siemdejong/sclicom/commit/df2974e41d371d964d7abf6ecb1d9a9ef7f7093c))


## v2.1.5 (2023-02-26)

### Ci

* ci: use conda shell ([`65d7e00`](https://github.com/siemdejong/sclicom/commit/65d7e00f52e5bfb1ae1162da05138c034d5db4c7))

* ci: fix conda python version ([`9ab509d`](https://github.com/siemdejong/sclicom/commit/9ab509d991eedd3e7019c0c9740032ef3bf254ea))

* ci: deps and release needs test ([`fe5e2ed`](https://github.com/siemdejong/sclicom/commit/fe5e2ed8472b18207b7f0cc8efe94983ce8de22a))

### Fix

* fix(tests): add tests for create_splits ([`780ed07`](https://github.com/siemdejong/sclicom/commit/780ed077000f55c0f7b86888c1f9646ec0085864))

### Unknown

* Merge branch &#39;main&#39; of github.com:siemdejong/dpat ([`c015208`](https://github.com/siemdejong/sclicom/commit/c01520824d0c279d847e0f1d21a27f3278822de8))


## v2.1.4 (2023-02-26)

### Fix

* fix(tests): add convert tests ([`819e23d`](https://github.com/siemdejong/sclicom/commit/819e23dbdb36824ffbcf5a520e8d43ac42acb0d8))


## v2.1.3 (2023-02-25)

### Fix

* fix(semantic-release): add build command

setup.py is not available, so python setup.py ... does not work.
build_command installs &#39;build&#39; en builds distributions. ([`c35a40f`](https://github.com/siemdejong/sclicom/commit/c35a40f214390c36a4f377523fc03560f3eb8cfd))


## v2.1.2 (2023-02-25)

### Fix

* fix(typing): add typing ([`0842497`](https://github.com/siemdejong/sclicom/commit/0842497f3498bf8f66af5ea917b5ecf012406b51))


## v2.1.1 (2023-02-23)

### Ci

* ci: update psr repo name ([`4ccf2e8`](https://github.com/siemdejong/sclicom/commit/4ccf2e8d35c166b8d9c4269b50b79f1b04ae1e54))

### Fix

* fix(swav): make swav return a number

Fix #7

SwAV by Lightly apparently needs the user to not use a datamodule.
If using the datamodule, call setup() on it and use
datamodule.x_dataloader() to pass to trainer.fit along with the model. ([`d21dcce`](https://github.com/siemdejong/sclicom/commit/d21dcce5c0590d94c87478b7c7eb8324476328f4))

* fix(deps): add tensorboard to dependencies ([`5ee3528`](https://github.com/siemdejong/sclicom/commit/5ee35280bf5d7e990bcf3c26ae13975e5990b782))

* fix(convert): only log skip if skip_count&gt;0 ([`0a58791`](https://github.com/siemdejong/sclicom/commit/0a587910b9f2fa7b32d05c21aa8b0c3facb671b7))


## v2.1.0 (2023-02-22)

### Ci

* ci: add workflow dispatch ([`10da34a`](https://github.com/siemdejong/sclicom/commit/10da34a559aeafd43055b210e744f3be7ae82b9e))

* ci: docformat, no-opt, print, ssort, black, flake8 ([`50e8ef4`](https://github.com/siemdejong/sclicom/commit/50e8ef4c9249cb03e71bfad8d589b625cf9ffc11))

* ci: add continuous integration

Move setup.py to pyproject.toml.
Add semantic-release github ci.
Add flake8 and yesqa to pre-commit. ([`81e50a0`](https://github.com/siemdejong/sclicom/commit/81e50a08299dee8ceb7754e4adcf45943dbd3cf7))

### Feature

* feat(data): add mean and std calculator ([`e6549f6`](https://github.com/siemdejong/sclicom/commit/e6549f681bd0f4500077dd2365afd5a9a231c972))

* feat(stage1): Implement swav with pl and lightly ([`3dcbdca`](https://github.com/siemdejong/sclicom/commit/3dcbdcab5c4de8ebd08addb8c0f2ce02fdfa4803))

### Fix

* fix(package): add dependencies ([`b3d340c`](https://github.com/siemdejong/sclicom/commit/b3d340c046929d3b06f417dc2ae40e61ac173e13))

* fix(package): package dpat subfolder ([`2fb78ed`](https://github.com/siemdejong/sclicom/commit/2fb78ed751908f28a0ca1719018c8a74d114b956))

### Unknown

* Update issue templates ([`7eb8142`](https://github.com/siemdejong/sclicom/commit/7eb8142a481dfaba4a61cb6d82e66d287579bbcf))


## v2.0.0 (2023-02-16)

### Breaking

* feat(installation): remove the need for config.yml

To install dpat in a script to use as api on windows, it is able
to achieve that with
import dpat
dpat.install_windows(&#34;path/to/vips/bin&#34;)

BREAKING CHANGE: installation via the config.yml is no longer possible.
It is also no longer needed for splits/convert cli operations.
For coming deep learning cli applications, it will be needed to fetch
the path to vipsbin from a config with deep learning options. ([`bdbed64`](https://github.com/siemdejong/sclicom/commit/bdbed641081d0717d97876aa3a76bb0a9f0c216f))

* fix(logging): let the cli configure logging

This allows for a user using the library to configure logging, e.g. use
`logging.getLogger(&#39;dpat&#39;).propagate = False`, if logging is not needed.
The cli will always log to the terminal.

BREAKING CHANGE: logging with the config file is now unsupported.
Configure logging in the application using the library. ([`85875fd`](https://github.com/siemdejong/sclicom/commit/85875fd6684933dfc01431be8f007c45b266e0ca))

### Documentation

* docs(cuda): add docs about cuda ([`58ca68f`](https://github.com/siemdejong/sclicom/commit/58ca68f97c7c6a956654a3f5cb2a6b694a5c2fdc))

* docs(logging): remove config.yml logging

Previous commit disabled logging config with config.yml. ([`60e6d90`](https://github.com/siemdejong/sclicom/commit/60e6d906e27db69203db389d268ca6e846973db1))

* docs(readme): clarify log/vips config

Add examples for `config.yml` for logging.
Show how to turn off logging propagation when using as a library. ([`0e06f66`](https://github.com/siemdejong/sclicom/commit/0e06f66875dd5aa3344106d062aeb935f1ab1f02))

### Fix

* fix(dpat): only read config.yml if windows

Linux users should be able to install vips with conda. ([`f823ccc`](https://github.com/siemdejong/sclicom/commit/f823ccca59c966622464f821da158a70e3449469))


## v1.4.0 (2023-02-15)

### Documentation

* docs(convert): change bulk to batch ([`2272355`](https://github.com/siemdejong/sclicom/commit/2272355593294d6b8963616dc02f295b44589be3))

### Feature

* feat(logging): add logging and log config

Logging is done to the NullHandler.
However, the user can turn logging on by setting the handler and
level in config.yml, like
LOGGING:
  handler: StreamHandler
  level: INFO ([`6c90359`](https://github.com/siemdejong/sclicom/commit/6c90359a5d9153555a1c12a9277d6869ce113525))

### Fix

* fix(cli): remove unnecessary __name__=__main__ chk ([`c5f5223`](https://github.com/siemdejong/sclicom/commit/c5f5223c4b1a6a5f8432f0406665a5fca35104d6))


## v1.3.0 (2023-02-15)

### Feature

* feat(dataset): add PMCHHGImageDataset

Add a dataset to loop through all tiles create from images
in an image directory. ([`1c65f85`](https://github.com/siemdejong/sclicom/commit/1c65f85f4e170171e27c6a104d2a82c9f0771149))


## v1.2.2 (2023-02-15)

### Fix

* fix(splits): change saved filename to relative

Fixes #1. ([`5ba6ed9`](https://github.com/siemdejong/sclicom/commit/5ba6ed95ce33f92d4a9677af37c1645944b75edb))


## v1.2.1 (2023-02-15)

### Documentation

* docs(cli): update docs of cli ([`b917e52`](https://github.com/siemdejong/sclicom/commit/b917e52856c06a0dbbb094a53d591a971215e150))

### Fix

* fix(convert): decompressionbomberror help ([`9ff923d`](https://github.com/siemdejong/sclicom/commit/9ff923db4ec297c78daac5e3c8fb294e42072983))


## v1.2.0 (2023-02-15)

### Feature

* feat(cli): change from argeparse to click ([`470ae65`](https://github.com/siemdejong/sclicom/commit/470ae65c54d90bc69cf93f764b3445c94433254d))


## v1.1.1 (2023-02-14)

### Fix

* fix(splits): a bug with default include/exclude

Argparse&#39;s action &#39;append&#39; appends arguments to the default.
Postprocessing the arguments fixes this. ([`3f46207`](https://github.com/siemdejong/sclicom/commit/3f462075fa6a07adef252454a9069a478e7414d3))


## v1.1.0 (2023-02-14)

### Documentation

* docs(readme): change installation heading ([`dd1416d`](https://github.com/siemdejong/sclicom/commit/dd1416df319638a642d1d28884c1fd34c81c9bb1))

### Feature

* feat(splits): change split logic

Create 5 stratified train-test folds. For every fold, randomly create
5 subfolds, dividing the training set into train-val (0.8/0.2).
Save every filename of every fold to a file with a unique filename,
denoting the set type, and subfold and fold id. ([`5f38156`](https://github.com/siemdejong/sclicom/commit/5f38156574a27694cd8d85f00f356bd335b50764))

* feat(splits): add inclusion and exclusion pattern

Some images might be needed to exclude by filename.
E.g. images with 300fast inside it.
Otherwise, only images with 200slow may be included. ([`caa5d2a`](https://github.com/siemdejong/sclicom/commit/caa5d2a3d17f01c738dcb6829f0f233cd0cc68cd))

* feat(splits): add overwrite argument ([`f2a664f`](https://github.com/siemdejong/sclicom/commit/f2a664fdeb3c36495570aa452ca74a378861e9f1))


## v1.0.0 (2023-02-14)

### Breaking

* feat(dpat): change project.ini to config.yml

YAML files are better structured.

BREAKING CHANGE: project.ini will not be read anymore.
PATHS.vips has to be set in config.yml, somewhere near the root
of the repository. ([`f13ed4b`](https://github.com/siemdejong/sclicom/commit/f13ed4bcd9e44d7cbbfe74443effc8ff4347d38a))

### Documentation

* docs(installation): clarify where to specify vipsbin ([`643009a`](https://github.com/siemdejong/sclicom/commit/643009ad8f01e10da84167df8792dc4baabcc160))

* docs(splits): add help of splits cli to README ([`e3ef258`](https://github.com/siemdejong/sclicom/commit/e3ef25843576de0b9313bb58462d073a0b258a43))

### Unknown

* Create LICENSE ([`d9f55dd`](https://github.com/siemdejong/sclicom/commit/d9f55dd719438612596f1a11398880c5cc0fbae3))


## v0.4.1 (2023-02-14)

### Fix

* fix(dpat): import pyvips before openslide ([`2d25ae0`](https://github.com/siemdejong/sclicom/commit/2d25ae00725775c880f7d3c568985f2a76b0c894))


## v0.4.0 (2023-02-14)

### Feature

* feat: add pre-commit

Make sure black and isort are ran. ([`620dccb`](https://github.com/siemdejong/sclicom/commit/620dccba985db8ab86ae6a6ea3fec3775b9aa057))

* feat(splits): create splits and run isort/black ([`fe8cf25`](https://github.com/siemdejong/sclicom/commit/fe8cf25ac3940ea1810b66edea92bdf080eba792))


## v0.3.0 (2023-02-13)

### Documentation

* docs: fix typo ([`18971e2`](https://github.com/siemdejong/sclicom/commit/18971e22813b39c2b3cb0679918e0f152f334e75))

* docs: fix 300fast typo ([`f0ed148`](https://github.com/siemdejong/sclicom/commit/f0ed148fc62ad89d387fece03112bcf64c3cb383))

* docs: clarify image fn needs scanprogram for tif ([`f0f7384`](https://github.com/siemdejong/sclicom/commit/f0f7384e8d548628b882099b2c3faebd380b7cd8))

* docs: clarify conda dependency ([`932569f`](https://github.com/siemdejong/sclicom/commit/932569f78a459c218366a1d2ee3d6127a1a3323c))

### Feature

* feat: add num-workers and #chunks to cli ([`0f322c4`](https://github.com/siemdejong/sclicom/commit/0f322c49bb3cbdaa7ecf961d85bb5c0c44f756f0))


## v0.2.0 (2023-02-13)

### Documentation

* docs: remove logo ([`f932b71`](https://github.com/siemdejong/sclicom/commit/f932b714653002ea6b8023d1eb4f49032e90e3ae))

* docs: make package and add readme

Add docs on how to build the package.
Add README.md to document usage of bulk convert. ([`e5fd4a5`](https://github.com/siemdejong/sclicom/commit/e5fd4a5c44c160430e7245bc80b185300e6379bb))

### Feature

* feat: clarify dlup will be installed with dpat ([`c062884`](https://github.com/siemdejong/sclicom/commit/c062884fcd7232db34c8b48a3c72bd06be095589))


## v0.1.0 (2023-02-13)

### Feature

* feat: initial commit

Add dlup and its dependencies, resolved for Windows.
Add any (e.g. bmp) to TIFF converter. ([`5d2fc3a`](https://github.com/siemdejong/sclicom/commit/5d2fc3a4371298d8de84eb296a735c80b55980f3))
