#!/usr/bin/env groovy

def dirPrefix = 'reframe-ci'
def loginBash = '#!/bin/bash -l'
def bashScript = 'ci-scripts/ci-runner.bash'
def machinesList = ['daint', 'dom', 'kesch', 'monch']
def machinesToRun = machinesList
def uniqueID

stage('Initialization') {
    node('master') {
        try {
            uniqueID = "${env.ghprbActualCommit[0..6]}-${env.BUILD_ID}"
            echo 'Environment Variables:'
            echo sh(script: 'env|sort', returnStdout: true)

            def githubComment = env.ghprbCommentBody
            if (githubComment == 'null') {
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
        } catch(err) {
            println err.toString()
            if (err.toString().contains('exit code 143')) {
                currentBuild.result = "ABORTED"
            }
            else if (err.toString().contains('Queue task was cancelled')) {
                currentBuild.result = "ABORTED"
            }
            else {
                currentBuild.result = "FAILURE"
            }
        }
    }
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
                def moduleDefinition = ''
                if (machineName == 'leone') {
                    moduleDefinition = '''module() { eval `/usr/bin/modulecmd bash $*`; }
                                          export -f module'''
                }
                dir(reframeDir) {
                    checkout scm
                    sh("""${loginBash}
                          ${moduleDefinition}
                          bash ${reframeDir}/${bashScript} -f ${reframeDir} -i ''""")
                }
            }
        }
    }

    try {
        parallel builds
        currentBuild.result = "SUCCESS"
    } catch(err) {
        if (err.toString().contains('exit code 143')) {
            currentBuild.result = "ABORTED"
            println "The Unittest was cancelled. Aborting....."
        }
        else if (err.toString().contains('Queue task was cancelled')) {
            currentBuild.result = "ABORTED"
            println "The Queue task was cancelled. Aborting....."
        }
        else {
            currentBuild.result = "FAILURE"
            println "The Unittest failed. Exiting....."
        }
    }
}

stage('Public Test') {
    if (currentBuild.result != 'SUCCESS') {
        println 'Not executing "Public Test" Stage'
        return
    }
    else {
        try {
            if (!('dom' in machinesToRun)) {
                return
            }
            node('dom') {
                def scratch = sh(returnStdout: true,
                             script: """${loginBash}
                                        echo \$SCRATCH""").trim()
                def reframeDir = "${scratch}/${dirPrefix}-dom-${uniqueID}"
                dir(reframeDir) {
                    sh("""${loginBash}
                          bash ${reframeDir}/$bashScript -f ${reframeDir} -i '' -g""")
                }
            }
            currentBuild.result = "SUCCESS"
        } catch(err) {
            if (err.toString().contains('exit code 143')) {
                currentBuild.result = "ABORTED"
                println "The Public Test was cancelled. Aborting....."
            }
            else if (err.toString().contains('Queue task was cancelled')) {
                currentBuild.result = "ABORTED"
                println "The Queue task was cancelled. Aborting....."
            }
            else {
                currentBuild.result = "FAILURE"
                println "The Public Test failed. Exiting....."
            }
        }
    }
}

builds = [:]
stage('Tutorial Check') {
    if (currentBuild.result != 'SUCCESS') {
        println 'Not executing "Tutorial Check" Stage'
        return
    }
    else {
        try {
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
            currentBuild.result = "SUCCESS"
        } catch(err) {
            if (err.toString().contains('exit code 143')) {
                currentBuild.result = "ABORTED"
                println "The Tutorial Check was cancelled. Aborting....."
            }
            else if (err.toString().contains('Queue task was cancelled')) {
                currentBuild.result = "ABORTED"
                println "The Queue task was cancelled. Aborting....."
            }
            else {
                currentBuild.result = "FAILURE"
                println "The Tutorial Check failed. Exiting....."
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
        try {
            parallel builds
            currentBuild.result = "SUCCESS"
        } catch(err) {
            if (err.toString().contains('exit code 143')) {
                currentBuild.result = "ABORTED"
                println "The Cleanup was cancelled. Aborting....."
            }
            else if (err.toString().contains('Queue task was cancelled')) {
                currentBuild.result = "ABORTED"
                println "The Queue task was cancelled. Aborting....."
            }
            else {
                currentBuild.result = "FAILURE"
                println "The Cleanup failed. Exiting....."
            }
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

    try {
        parallel builds
        currentBuild.result = "SUCCESS"
    } catch(err) {
        if (err.toString().contains('exit code 143')) {
            currentBuild.result = "ABORTED"
            println "The Build step was cancelled. Aborting....."
        }
        else if (err.toString().contains('Queue task was cancelled')) {
            currentBuild.result = "ABORTED"
            println "The Queue task was cancelled. Aborting....."
        }
        else {
            currentBuild.result = "FAILURE"
            println "The Build step failed. Exiting....."
        }
    }
}
