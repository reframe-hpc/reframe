#!/usr/bin/env groovy

/**
* Checks if as message contains a given command and machine.
*
* @param message The actual message.
* @param command The command.
* @param machine The machine.
* @return A boolean indicating whether there is a match.
*/
boolean machineCheck(String message, String command, String machine) {
    def matchPattern = "@jenkins-cscs\\s+${command}\\s+${machine}"
    return message ==~ machinePattern
}

def dirPrefix = 'reframe-ci'
def loginBash = '#!/bin/bash -l'
def bashScript = 'ci-scripts/ci-runner.bash'
def publicSettings = 'config/generic.py'
def cscsSettings = 'config/cscs.py'
def machinesList = ['daint', 'dom', 'kesch', 'leone', 'monch']
def githubComment = env.ghprbCommentBody
def machinesToRun = machinesList
def shortCommit

stage('Initialization') {
    node('master') {
        try {
            def scmVars = checkout scm
            shortCommit = scmVars.GIT_COMMIT[0..6]
            currentBuild.result = 'SUCCESS'
        } catch(err) {
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
    println 'Initialization failed. Exiting...'
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
                def reframeDir = "${scratch}/${dirPrefix}-${machineName}-${shortCommit}"
                def moduleDefinition = ''
                if (machineName == 'leone')
                    moduleDefinition = '''module() { eval `/usr/bin/modulecmd bash $*`; }
                                          export -f module'''
                dir(reframeDir) {
                    checkout scm
                    sh("""${loginBash}
                          ${moduleDefinition}
                          cp reframe/settings.py ${publicSettings}
                          cp ${cscsSettings} reframe/settings.py
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
            node('dom') {
                def scratch = sh(returnStdout: true,
                             script: """${loginBash}
                                        echo \$SCRATCH""").trim()
                def reframeDir = "${scratch}/${dirPrefix}-dom-${shortCommit}"
                dir(reframeDir) {
                    sh("""${loginBash}
                          bash ${reframeDir}/$bashScript -f ${reframeDir} -i '' -p""")
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
            node('daint') {
                def scratch = sh(returnStdout: true,
                                 script: """${loginBash}
                                            echo \$SCRATCH""").trim()
                def reframeDir = "${scratch}/${dirPrefix}-daint-${shortCommit}"
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
                    def reframeDir = "${scratch}/${dirPrefix}-${machineName}-${shortCommit}"
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

def staleCleanupInterval = 1
builds = [:]
stage('Cleanup Stale') {
     for (mach in machinesList) {
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
