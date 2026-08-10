//
//  main.swift
//  AudioToolBenchmarkCLI
//
//  `audio-tool-bench`: what every model in this package costs, on this machine.
//
//  Build: xcodebuild build -scheme audio-tool-bench -configuration Release \
//           -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
//  Run:   .build/DerivedData/Build/Products/Release/audio-tool-bench --list
//
//  It has to run from the products directory, for the same reason `audio-tool`
//  does: that is where MLX's metallib is. See AGENTS.md, "Metal/MLX".
//
//  Kept to argument parsing and dispatch. Everything a function needs is passed
//  in rather than read from here, because top-level variables in main.swift are
//  main-actor isolated under Swift 6 and reaching for them from the harness would
//  drag that isolation into code that has no reason to want it.
//

import Foundation
import AudioToolBenchmark

let parsedOptions: BenchmarkOptions
do {
    parsedOptions = try BenchmarkOptions.parse(CommandLine.arguments)
} catch {
    FileHandle.standardError.write(Data("\(error)\n\n".utf8))
    print(usage)
    exit(64)  // EX_USAGE
}

if parsedOptions.listOnly {
    BenchmarkDriver.list(parsedOptions)
    exit(0)
}

// Top-level code cannot await and the whole harness is async, so the work goes in
// a Task and the process is held open by the run loop. Same shape as
// `audio-tool`'s main.swift.
Task { [options = parsedOptions] in
    let status: Int32
    if let caseID = options.childCaseID {
        status = await BenchmarkDriver.runSingleCase(caseID, options: options)
    } else if options.prefetchOnly {
        status = await BenchmarkDriver.prefetchAll(options)
    } else {
        status = await BenchmarkDriver.runAll(options)
    }
    exit(status)
}

RunLoop.main.run()
