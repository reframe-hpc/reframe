==================
Welcome to ReFrame
==================

ReFrame is a powerful framework for writing system regression tests and benchmarks, specifically targeted to HPC systems.
The goal of the framework is to abstract away the complexity of the interactions with the system, separating the logic of a test from the low-level details, which pertain to the system configuration and setup.
This allows users to write portable tests in a declarative way that describes only the test's functionality.

Tests in ReFrame are simple Python classes that specify the basic variables and parameters of the test.
ReFrame offers an intuitive and very powerful syntax that allows users to create test libraries, test factories, as well as complete test workflows using other tests as fixtures.
ReFrame will load the tests and send them down a well-defined pipeline that will execute them in parallel.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

ReFrame also offers a high-level and flexible abstraction for writing sanity and performance checks for your regression tests, without having to care about the details of parsing output files, searching for patterns and testing against reference values for different systems.

Finally, ReFrame offers a powerful and efficient runtime for running and managing the execution of tests, as well as integration with common logging facilities, where ReFrame can send live data from currently running performance tests.

Publications
============

Presentations & Talks
---------------------

* Slides [`pdf <https://users.ugent.be/~kehoste/eum24/008_eum24_ReFrame.pdf>`__][`talk <https://www.youtube.com/live/uSEeU-VJf6k?si=YB19mFpG6aAEBOgH>`__] @ `9th EasyBuild User Meeting 2024 <https://easybuild.io/eum24/>`__.
* Slides [`part 1 <https://docs.google.com/presentation/d/1GmO2Uf29SaLg36bPB9g9eeaKMN-bLlDJ5IvLGLQJfD8/edit?usp=share_link>`__][`part 2 <https://drive.google.com/file/d/1gZwch0BPc1wDEkMwbM4vxCpMzWIx-Lo1/view?usp=sharing>`__][`talk <https://youtu.be/0ApEKc185Bw>`__] @ `8th EasyBuild User Meeting 2023 <https://easybuild.io/eum23/>`__.
* Slides [`pdf <https://drive.google.com/file/d/1vmaWyRHgtq3DrYhSCVBzR8U5ErKbxGNf/view?usp=sharing>`__] @ `7th EasyBuild User Meeting 2022 <https://easybuild.io/eum22/>`__.
* Slides [`pdf <https://drive.google.com/file/d/1kNZu1QNBDDsbKarzwNWYjTGKgOukg-96/view?usp=sharing>`__] @ `6th EasyBuild User Meeting 2021 <https://easybuild.io/eum21/>`__.
* Slides [`pdf <https://drive.google.com/open?id=1W7R5lfRkXvBpVDSZ7dVBadk_d3K4dFrS>`__] @ `5th EasyBuild User Meeting 2020 <https://github.com/easybuilders/easybuild/wiki/5th-EasyBuild-User-Meeting>`__.
* Slides [`pdf <https://drive.google.com/open?id=1Z3faPh9OSSXvlLHL07co3MRRn443dYsY>`__] @ `HPC System Testing BoF <https://sc19.supercomputing.org/session/?sess=sess324>`__, SC'19.
* Slides [`pdf <https://drive.google.com/open?id=1JOFqY3ejbR1X5kTn_IZyp1GlCd2ZZS58>`__] @ `HUST 2019 <https://sc19.supercomputing.org/session/?sess=sess116>`__, SC'19.
* Slides [`pdf <https://drive.google.com/open?id=1iwg1I48LVaWhhZCZIYPJSi3hdFLRcuhi>`__] @ `HPC Knowledge Meeting '19 <https://hpckp.org/>`__.
* Slides [`pdf <https://fosdem.org/2019/schedule/event/reframe/attachments/slides/3226/export/events/attachments/reframe/slides/3226/FOSDEM_2019.pdf>`__] & `Talk <https://fosdem.org/2019/schedule/event/reframe/>`__ @ `FOSDEM'19 <https://fosdem.org/2019/>`__.
* Slides [`pdf <https://indico.cism.ucl.ac.be/event/4/contributions/24/attachments/30/62/ReFrame_EUM_2019.pdf>`__] @ `4th EasyBuild User Meeting <https://github.com/easybuilders/easybuild/wiki/4th-EasyBuild-User-Meeting>`__.
* Slides [`pdf <https://drive.google.com/open?id=1bSykDrl1e2gPflf4jFJ8kfe_SZAtrJ_Q>`__] @ `HUST 2018 <https://sc18.supercomputing.org/>`__, SC'18.
* Slides [`pdf <https://github.com/eth-cscs/UserLabDay/blob/master/2018/slides/ci_and_regression/ReFrame_CI.pdf>`__] @ `CSCS User Lab Day 2018 <https://github.com/eth-cscs/UserLabDay>`__.
* Slides [`pdf <https://drive.google.com/open?id=1sZhibvUlGlT670aOHPdMlWFffWptYzLX>`__] @ `HPC Advisory Council 2018 <http://www.hpcadvisorycouncil.com/events/2018/swiss-workshop/>`__.
* Slides [`pdf <https://drive.google.com/open?id=1EyJ-siupkgLeVT54A4WlFpQtrJaU0xOy>`__] @ `SC17 <https://sc17.supercomputing.org/>`__.
* Slides [`pdf <https://cug.org/proceedings/cug2017_proceedings/includes/files/pap122s2-file2.pdf>`__] @ `CUG 2017 <https://cug.org/cug-2017/>`__.


Webinars & Tutorials
--------------------

* "ReFrame – Efficient System and Application Performance Testing," CSCS Webinar, Aug. 29, 2022 [`slides <https://drive.google.com/file/d/1nOS_daleR79ZB1IaToVdW5mDpJQYRcY2/view?usp=sharing>`__] [`recording <https://youtu.be/NDxlKATEcQk>`__] [`demo run <https://asciinema.org/a/517693>`__].
* Tutorial at 6th EasyBuild User Meeting 2021 [`YouTube <https://youtube.com/playlist?list=PLhnGtSmEGEQjySVEPTUSLpewpOWwX5mjb>`__]


Papers
------

- Vasileios Karakasis et al. "A regression framework for checking the health of large HPC systems". In: *Cray User Group 2017* (Redmond, Washington, USA, May 8--11, 2017). [`pdf <https://cug.org/proceedings/cug2017_proceedings/includes/files/pap122s2-file1.pdf>`__]

- Vasileios Karakasis et al. "Enabling Continuous Testing of HPC Systems Using ReFrame". In: *Tools and Techniques for High Performance Computing. HUST -- Annual Workshop on HPC User Support Tools* (Denver, Colorado, USA, Nov. 17--18, 2019). Ed. by Guido Juckeland and Sunita Chandrasekaran. Vol. 1190. Communications in Computer and Information Science. Cham, Switzerland: Springer International Publishing, Mar. 2020, pp. 49--68. isbn: 978-3-030-44728-1. doi: `10.1007/978-3-030-44728-1_3 <https://doi.org/10.1007/978-3-030-44728-1_3>`__.


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   started
   tutorial
   howto
   topics
   manuals
   whats_new_40
   hpctestlib
