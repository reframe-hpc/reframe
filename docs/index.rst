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

Additionally, ReFrame supports storing the test results in a database allowing for later inspection, basic analytics and performance comparisons.

Finally, ReFrame offers a powerful and efficient runtime for running and managing the execution of tests, as well as integration with common logging facilities, where ReFrame can send live data from currently running performance tests.

Publications
============

Presentations & Talks
---------------------

* [`slides + talk <https://fosdem.org/2025/schedule/event/fosdem-2025-4755-adding-built-in-support-for-basic-performance-test-analytics-to-reframe/>`__] "Adding built-in support for basic performance test analytics to ReFrame," `FOSDEM 25 <https://fosdem.org/2025/>`__.
* [`slides <https://drive.google.com/file/d/1cwlIipJtJoD-0xGcDMxbL4FQe3xtwAen/view?usp=sharing>`__] "Introduction to ReFrame," CINECA visit, Jun 2024.
* [`slides <https://users.ugent.be/~kehoste/eum24/008_eum24_ReFrame.pdf>`__][`recording <https://www.youtube.com/live/uSEeU-VJf6k?si=YB19mFpG6aAEBOgH>`__] "Recent Advances in ReFrame," `9th EasyBuild User Meeting 2024 <https://easybuild.io/eum24/>`__.
* [`slides <https://docs.google.com/presentation/d/1GmO2Uf29SaLg36bPB9g9eeaKMN-bLlDJ5IvLGLQJfD8/edit?usp=share_link>`__][`recording <https://youtu.be/0ApEKc185Bw>`__] "Recent Advances in ReFrame," `8th EasyBuild User Meeting 2023 <https://easybuild.io/eum23/>`__.
* [`slides <https://drive.google.com/file/d/1gZwch0BPc1wDEkMwbM4vxCpMzWIx-Lo1/view?usp=sharing>`__][`recording <https://youtu.be/0ApEKc185Bw?si=uxa_3QhMjTagS2to&t=1485>`__] "Embracing ReFrame Programmable Configurations," `8th EasyBuild User Meeting 2023 <https://easybuild.io/eum23/>`__.
* [`slides <https://drive.google.com/file/d/1vmaWyRHgtq3DrYhSCVBzR8U5ErKbxGNf/view?usp=sharing>`__] "ReFrame Update," `7th EasyBuild User Meeting 2022 <https://easybuild.io/eum22/>`__.
* [`slides <https://drive.google.com/file/d/1kNZu1QNBDDsbKarzwNWYjTGKgOukg-96/view?usp=sharing>`__] "Writing powerful HPC regression tests with ReFrame," `6th EasyBuild User Meeting 2021 <https://easybuild.io/eum21/.>`__
* [`slides <https://drive.google.com/open?id=1W7R5lfRkXvBpVDSZ7dVBadk_d3K4dFrS>`__] "ReFrame: A Framework for Writing Regression Tests for HPC Systems," `5th EasyBuild User Meeting 2020 <https://github.com/easybuilders/easybuild/wiki/5th-EasyBuild-User-Meeting>`__.
* [`slides <https://drive.google.com/open?id=1Z3faPh9OSSXvlLHL07co3MRRn443dYsY>`__] "Enabling Continuous Testing of HPC Systems using ReFrame," `HPC System Testing BoF <https://sc19.supercomputing.org/session/?sess=sess324>`__, SC'19.
* [`slides <https://drive.google.com/open?id=1JOFqY3ejbR1X5kTn_IZyp1GlCd2ZZS58>`__] "Enabling Continuous Testing of HPC Systems using ReFrame," `HUST 2019 <https://sc19.supercomputing.org/session/?sess=sess116>`__, SC'19.
* [`slides <https://drive.google.com/open?id=1iwg1I48LVaWhhZCZIYPJSi3hdFLRcuhi>`__] "ReFrame: A Tool for Enabling Regression Testing and Continuous Integration for HPC Systems," `HPC Knowledge Meeting '19 <https://hpckp.org/>`__.
* [`slides <https://fosdem.org/2019/schedule/event/reframe/attachments/slides/3226/export/events/attachments/reframe/slides/3226/FOSDEM_2019.pdf>`__][`recording <https://fosdem.org/2019/schedule/event/reframe/>`__] "ReFrame: A Regression Testing and Continuous Integration Framework for HPC systems," `FOSDEM'19 <https://fosdem.org/2019/>`__.
* [`slides <https://indico.cism.ucl.ac.be/event/4/contributions/24/attachments/30/62/ReFrame_EUM_2019.pdf>`__] "ReFrame: A Regression Testing and Continuous Integration Framework for HPC systems," `4th EasyBuild User Meeting <https://github.com/easybuilders/easybuild/wiki/4th-EasyBuild-User-Meeting>`__.
* [`slides <https://drive.google.com/open?id=1bSykDrl1e2gPflf4jFJ8kfe_SZAtrJ_Q>`__] "ReFrame: A Regression Testing and Continuous Integration Framework for HPC systems," `HUST 2018 <https://sc18.supercomputing.org/>`__, SC'18.
* [`slides <https://github.com/eth-cscs/UserLabDay/blob/master/2018/slides/ci_and_regression/ReFrame_CI.pdf>`__] "Regression Testing and Continuous Integration with ReFrame," `CSCS User Lab Day 2018 <https://github.com/eth-cscs/UserLabDay>`__.
* [`slides <https://drive.google.com/open?id=1sZhibvUlGlT670aOHPdMlWFffWptYzLX>`__] "ReFrame: A Regression Testing Framework Enabling Continuous Integration of Large HPC Systems," `HPC Advisory Council 2018 <http://www.hpcadvisorycouncil.com/events/2018/swiss-workshop/>`__.
* [`slides <https://drive.google.com/open?id=1EyJ-siupkgLeVT54A4WlFpQtrJaU0xOy>`__] "ReFrame: A Regression Testing Tool for HPC Systems," Regression testing BoF, `SC17 <https://sc17.supercomputing.org/>`__.
* [`slides <https://cug.org/proceedings/cug2017_proceedings/includes/files/pap122s2-file2.pdf>`__] "ReFrame: A regression framework for checking the health of large HPC systems" `CUG 2017 <https://cug.org/cug-2017/>`__.


Webinars & Tutorials
--------------------

* [`slides <https://drive.google.com/file/d/1nOS_daleR79ZB1IaToVdW5mDpJQYRcY2/view?usp=sharing>`__][`recording <https://youtu.be/NDxlKATEcQk>`__][`demo run <https://asciinema.org/a/517693>`__] "ReFrame – Efficient System and Application Performance Testing," CSCS Webinar, Aug. 29, 2022.
* [`recording <https://youtube.com/playlist?list=PLhnGtSmEGEQjySVEPTUSLpewpOWwX5mjb>`__] "ReFrame Tutorial," 6th EasyBuild User Meeting 2021.


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
