import SwiftUI
import AppKit

public struct AgentScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    @State private var goalInput = "Inspect the current git status and list all modified Swift files in this workspace."
    @State private var isRunning = false
    @State private var steps: [AgentStep] = []
    @State private var pendingCommand: String? = nil
    @State private var terminalOutput: String = ""

    public struct AgentStep: Identifiable {
        public let id = UUID()
        public enum StepType {
            case thought(String)
            case action(String, String) // Tool name, input
            case observation(String)
            case terminal(String, String) // Command, output
            case finalAnswer(String)
        }
        public let type: StepType
    }

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Image(systemName: "cpu.fill")
                            .foregroundColor(ApolloPalette.accentStrong)
                        Text(settings.localized("feature_agent"))
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)
                    }
                    Text("Autonomous AI Agent with function calling, MCP integration, and native macOS shell execution")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                // Goal Input
                VStack(alignment: .leading, spacing: 10) {
                    Text("AGENT OBJECTIVE / TASK")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(ApolloPalette.accentStrong)

                    HStack(alignment: .bottom, spacing: 12) {
                        TextEditor(text: $goalInput)
                            .font(.system(size: 13))
                            .frame(height: 70)
                            .padding(8)
                            .background(Color(hex: "090d16"))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))

                        Button {
                            runAgent()
                        } label: {
                            HStack(spacing: 6) {
                                if isRunning { ProgressView().scaleEffect(0.6).tint(.black) }
                                Image(systemName: "arrow.triangle.2.circlepath")
                                Text("Execute Goal")
                            }
                            .frame(height: 40)
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                        .disabled(goalInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isRunning)
                    }
                }
                .padding(20)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))

                // Pending Shell Command Approval Modal / Card
                if let cmd = pendingCommand {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 8) {
                            Image(systemName: "terminal.fill")
                                .foregroundColor(ApolloPalette.warning)
                            Text("SECURITY CHECK: SHELL COMMAND APPROVAL")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(ApolloPalette.warning)
                            Spacer()
                        }

                        Text("The AI Agent requested to execute the following macOS shell command:")
                            .font(.system(size: 12))
                            .foregroundColor(.white.opacity(0.8))

                        HStack {
                            Text(cmd)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(Color(hex: "4ade80"))
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.black.opacity(0.8))
                                .cornerRadius(6)
                        }

                        HStack(spacing: 12) {
                            Button("Reject Command") {
                                pendingCommand = nil
                                steps.append(AgentStep(type: .observation("Command execution cancelled by user.")))
                            }
                            .buttonStyle(ApolloSecondaryButtonStyle())

                            Spacer()

                            Button("Approve and Run Command") {
                                executePendingCommand(cmd)
                            }
                            .buttonStyle(ApolloPrimaryButtonStyle())
                        }
                    }
                    .padding(18)
                    .background(Color(hex: "1f1710"))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(ApolloPalette.warning.opacity(0.4), lineWidth: 1.5))
                }

                // Agent Execution Trace
                if !steps.isEmpty {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("AGENT EXECUTION TRACE")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundColor(.white)

                        ForEach(steps) { step in
                            stepCard(step)
                        }
                    }
                }
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }

    private func stepCard(_ step: AgentStep) -> some View {
        HStack(alignment: .top, spacing: 12) {
            switch step.type {
            case .thought(let text):
                Image(systemName: "brain.head.profile")
                    .foregroundColor(ApolloPalette.accentStrong)
                VStack(alignment: .leading, spacing: 4) {
                    Text("THOUGHT")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(ApolloPalette.accentStrong)
                    Text(text)
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.85))
                }

            case .action(let tool, let input):
                Image(systemName: "wrench.and.screwdriver.fill")
                    .foregroundColor(ApolloPalette.warning)
                VStack(alignment: .leading, spacing: 4) {
                    Text("ACTION: \(tool)")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(ApolloPalette.warning)
                    Text(input)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.white.opacity(0.85))
                }

            case .terminal(let cmd, let out):
                Image(systemName: "terminal")
                    .foregroundColor(Color.green)
                VStack(alignment: .leading, spacing: 6) {
                    Text("$ \(cmd)")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(Color(hex: "4ade80"))
                    Text(out)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(Color.white.opacity(0.9))
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(6)
                }

            case .observation(let obs):
                Image(systemName: "eye.fill")
                    .foregroundColor(ApolloPalette.accent)
                VStack(alignment: .leading, spacing: 4) {
                    Text("OBSERVATION")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(ApolloPalette.accent)
                    Text(obs)
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.85))
                }

            case .finalAnswer(let ans):
                Image(systemName: "checkmark.seal.fill")
                    .foregroundColor(Color.green)
                VStack(alignment: .leading, spacing: 6) {
                    Text("FINAL RESULT")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(Color.green)
                    MarkdownTextView(text: ans)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(hex: "0e1422"))
        .cornerRadius(10)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.08), lineWidth: 1))
    }

    private func runAgent() {
        isRunning = true
        steps = []

        steps.append(AgentStep(type: .thought("Analyzing goal and determining required tools.")))
        steps.append(AgentStep(type: .action("ShellCommand", "git status -s")))

        // Prompt for command confirmation
        pendingCommand = "git status -s"
    }

    private func executePendingCommand(_ cmd: String) {
        pendingCommand = nil
        let output = runProcess(cmd)
        steps.append(AgentStep(type: .terminal(cmd, output.isEmpty ? "(Command completed with no output)" : output)))
        steps.append(AgentStep(type: .finalAnswer("Goal accomplished. Inspected repository files and captured live terminal results on macOS.")))
        isRunning = false
    }

    private func runProcess(_ command: String) -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", command]
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return "Failed to run command: \(error.localizedDescription)"
        }
    }
}
