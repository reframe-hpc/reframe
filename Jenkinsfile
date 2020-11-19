#!/usr/bin/env groovy

def dirPrefix = 'reframe-ci'
def loginBash = '#!/bin/bash -l'
def bashScript = 'ci-scripts/ci-runner.bash'
def machinesList = params.machines.split()
def machinesToRun = machinesList
def runTests = true
def uniqueID

stage('Initialization') {
    node('master') {
        catchError(stageResult: 'FAILURE') {
            uniqueID = "${env.ghprbActualCommit[0..6]}-${env.BUILD_ID}"
            echo 'Environment Variables:'
            echo sh(script: 'env|sort', returnStdout: true)

            def githubComment = env.ghprbCommentBody
            if (githubComment == 'null' || !githubComment.trim().startsWith('@jenkins-cscs')) {
                machinesToRun = machinesList
                currentBuild.result = 'SUCCESS'
                return
            }

            def splittedComment = githubComment.split()
            if (splittedComment.size() < 3) {
                println 'No machines were found. Aborting...'
                currentBuild.result = 'ABORTED'
                return
            }
            if (splittedComment[1] != 'retry') {
                println "Invalid command ${splittedComment[1]}. Aborting..."
                currentBuild.result = 'ABORTED'
                return
            }
            if (splittedComment[2] == 'all') {
                machinesToRun = machinesList
                currentBuild.result = 'SUCCESS'
                return
            }
            else if (splittedComment[2] == 'none') {
                runTests = false
                currentBuild.result = 'SUCCESS'
                return
            }

            machinesRequested = []
            for (i = 2; i < splittedComment.size(); i++) {
                machinesRequested.add(splittedComment[i])
            }

            machinesToRun = machinesRequested.findAll({it in machinesList})
            if (!machinesToRun) {
                println 'No machines were found. Aborting...'
                currentBuild.result = 'ABORTED'
                return
            }
            currentBuild.result = 'SUCCESS'
        }
    }
}

if (!runTests) {
    println "Won't execute any test (${currentBuild.result}). Exiting..."
    return
}

if (currentBuild.result != 'SUCCESS') {
    println "Initialization failed (${currentBuild.result}). Exiting..."
    return
}

def builds = [:]
stage('Unittest') {
    for (mach in machinesToRun) {
        def machineName = mach
        builds[machineName] = {
            node(machineName) {
                def scratch = sh(returnStdout: true,
                                 script: """${loginBash}
                                            echo \$SCRATCH""").trim()
                def reframeDir = "${scratch}/${dirPrefix}-${machineName}-${uniqueID}"
                dir(reframeDir) {
                    checkout scm
                    sh("""${loginBash}
                          bash ${reframeDir}/${bashScript} -f ${reframeDir} -i ''""")
                }
            }
        }
    }

    catchError(stageResult: 'FAILURE') {
        parallel builds
    }
}

builds = [:]
stage('Tutorial Check') {
    if (currentBuild.result != 'SUCCESS') {
        println 'Not executing "Tutorial Check" Stage'
        return
    }
    else {
        catchError(stageResult: 'FAILURE') {
            if (!('daint' in machinesToRun)) {
                return
            }
            node('daint') {
                def scratch = sh(returnStdout: true,
                                 script: """${loginBash}
                                            echo \$SCRATCH""").trim()
                def reframeDir = "${scratch}/${dirPrefix}-daint-${uniqueID}"
                dir(reframeDir) {
                    sh("""${loginBash}
                          bash ${reframeDir}/${bashScript} -f ${reframeDir} -i '' -t""")
                }
            }
        }
    }
}

builds = [:]
stage('Cleanup') {
    if (currentBuild.result != 'SUCCESS') {
        println 'Not executing "Cleanup" Stage'
        return
    }
    else {
        for (mach in machinesToRun) {
            def machineName = mach
            builds[machineName] = {
                node(machineName) {
                    def scratch = sh(returnStdout: true,
                                     script: """$loginBash
                                                echo \$SCRATCH""").trim()
                    def reframeDir = "${scratch}/${dirPrefix}-${machineName}-${uniqueID}"
                    sh("""${loginBash}
                          rm -rf ${reframeDir}
                          date""")

                }
            }
        }
        catchError(stageResult: 'FAILURE') {
            parallel builds
        }
    }
}

def staleCleanupInterval = 3
builds = [:]
stage('Cleanup Stale') {
     for (mach in machinesToRun) {
        def machineName = mach
        builds[machineName] = {
            node(machineName) {
                def scratch = sh(returnStdout: true,
                                 script: """${loginBash}
                                            echo \$SCRATCH""").trim()
                sh("""${loginBash}
                      find ${scratch} -maxdepth 1 -name 'reframe-ci*' -ctime +${staleCleanupInterval} -type d -exec printf 'Removing  %s\\n' {} +
                      find ${scratch} -maxdepth 1 -name 'reframe-ci*' -ctime +${staleCleanupInterval} -type d -exec rm -rf {} +""")
            }
        }
    }

    catchError(stageResult: 'FAILURE') {
        parallel builds
    }
}
