# Changelog

<!--next-version-placeholder-->

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
