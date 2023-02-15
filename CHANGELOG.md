# Changelog

<!--next-version-placeholder-->

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
