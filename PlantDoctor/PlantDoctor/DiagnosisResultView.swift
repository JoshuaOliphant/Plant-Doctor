//
//  DiagnosisResultView.swift
//  PlantDoctor
//
//  Display the diagnosis results in a readable, actionable format
//

import SwiftUI

struct DiagnosisResultView: View {
    let diagnosis: DiagnoseResponse
    let image: UIImage?
    let onDiagnoseAnother: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header with plant image
                HeaderSection(image: image, species: diagnosis.speciesGuess)

                // Issues found
                if !diagnosis.issues.isEmpty {
                    IssuesSection(issues: diagnosis.issues)
                }

                // Explanation
                ExplanationSection(explanation: diagnosis.explanation)

                // Immediate actions
                if !diagnosis.immediateActions.isEmpty {
                    ImmediateActionsSection(actions: diagnosis.immediateActions)
                }

                // Recovery plan
                RecoveryPlanSection(plan: diagnosis.recoveryPlan)

                // Confidence notes
                if !diagnosis.confidenceNotes.isEmpty {
                    ConfidenceNotesSection(notes: diagnosis.confidenceNotes)
                }

                // Stats footer
                StatsFooter(
                    processingTimeMs: diagnosis.processingTimeMs,
                    costCents: diagnosis.modelCostCents
                )

                // Action button
                Button(action: onDiagnoseAnother) {
                    Label("Diagnose Another Plant", systemImage: "camera.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                .padding(.top, 8)
            }
            .padding()
        }
        .navigationTitle("Diagnosis")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Header Section

struct HeaderSection: View {
    let image: UIImage?
    let species: SpeciesGuess

    var body: some View {
        VStack(spacing: 12) {
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(height: 200)
                    .clipped()
                    .cornerRadius(12)
            }

            VStack(spacing: 4) {
                Text(species.name)
                    .font(.title2)
                    .fontWeight(.bold)

                HStack {
                    Text("Confidence:")
                        .foregroundColor(.secondary)
                    Text("\(Int(species.confidence * 100))%")
                        .fontWeight(.medium)
                        .foregroundColor(confidenceColor(species.confidence))
                }
                .font(.subheadline)
            }
        }
    }

    private func confidenceColor(_ confidence: Double) -> Color {
        switch confidence {
        case 0.8...1.0: return .green
        case 0.5..<0.8: return .orange
        default: return .red
        }
    }
}

// MARK: - Issues Section

struct IssuesSection: View {
    let issues: [Issue]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Issues Found", icon: "exclamationmark.triangle.fill")

            ForEach(issues) { issue in
                IssueCard(issue: issue)
            }
        }
    }
}

struct IssueCard: View {
    let issue: Issue

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(issue.name)
                    .font(.headline)

                Spacer()

                SeverityBadge(severity: issue.severity)
            }

            // Confidence bar
            HStack {
                Text("Confidence:")
                    .font(.caption)
                    .foregroundColor(.secondary)

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .fill(Color(.systemGray5))
                            .frame(height: 6)
                            .cornerRadius(3)

                        Rectangle()
                            .fill(severityColor)
                            .frame(width: geometry.size.width * issue.confidence, height: 6)
                            .cornerRadius(3)
                    }
                }
                .frame(height: 6)

                Text("\(Int(issue.confidence * 100))%")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(width: 40, alignment: .trailing)
            }

            // Visual evidence
            if !issue.visualEvidence.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Evidence:")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    ForEach(issue.visualEvidence, id: \.self) { evidence in
                        HStack(alignment: .top, spacing: 6) {
                            Text("-")
                                .foregroundColor(.secondary)
                            Text(evidence)
                        }
                        .font(.caption)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var severityColor: Color {
        switch issue.severity.lowercased() {
        case "mild": return .green
        case "moderate": return .orange
        case "severe": return .red
        default: return .gray
        }
    }
}

struct SeverityBadge: View {
    let severity: String

    var body: some View {
        Text(severity.uppercased())
            .font(.caption2)
            .fontWeight(.bold)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(backgroundColor)
            .foregroundColor(.white)
            .cornerRadius(4)
    }

    private var backgroundColor: Color {
        switch severity.lowercased() {
        case "mild": return .green
        case "moderate": return .orange
        case "severe": return .red
        default: return .gray
        }
    }
}

// MARK: - Explanation Section

struct ExplanationSection: View {
    let explanation: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "What's Happening", icon: "lightbulb.fill")

            Text(explanation)
                .font(.body)
                .foregroundColor(.primary)
        }
    }
}

// MARK: - Immediate Actions Section

struct ImmediateActionsSection: View {
    let actions: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Do This Today", icon: "bolt.fill")

            ForEach(Array(actions.enumerated()), id: \.offset) { index, action in
                HStack(alignment: .top, spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 28, height: 28)

                        Text("\(index + 1)")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                    }

                    Text(action)
                        .font(.body)
                }
            }
        }
        .padding()
        .background(Color.green.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Recovery Plan Section

struct RecoveryPlanSection: View {
    let plan: RecoveryPlan

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                SectionHeader(title: "Recovery Plan", icon: "calendar")

                Spacer()

                VStack(alignment: .trailing) {
                    Text("\(plan.durationWeeks) weeks")
                        .font(.caption)
                        .fontWeight(.medium)

                    HStack(spacing: 4) {
                        Text("Prognosis:")
                            .foregroundColor(.secondary)
                        Text(plan.prognosis.capitalized)
                            .fontWeight(.medium)
                            .foregroundColor(prognosisColor)
                    }
                    .font(.caption)
                }
            }

            ForEach(plan.weeks) { week in
                WeekCard(week: week)
            }
        }
    }

    private var prognosisColor: Color {
        switch plan.prognosis.lowercased() {
        case "good": return .green
        case "fair": return .orange
        case "guarded": return .red
        default: return .gray
        }
    }
}

struct WeekCard: View {
    let week: WeekPlan

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Week \(week.week)")
                    .font(.subheadline)
                    .fontWeight(.bold)

                Text("-")
                    .foregroundColor(.secondary)

                Text(week.focus)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            ForEach(week.tasks, id: \.self) { task in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "circle")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.top, 4)

                    Text(task)
                        .font(.subheadline)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Confidence Notes Section

struct ConfidenceNotesSection: View {
    let notes: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "info.circle.fill")
                .foregroundColor(.blue)

            Text(notes)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.blue.opacity(0.1))
        .cornerRadius(8)
    }
}

// MARK: - Stats Footer

struct StatsFooter: View {
    let processingTimeMs: Int
    let costCents: Double

    var body: some View {
        HStack {
            Spacer()

            HStack(spacing: 16) {
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                    Text(formattedTime)
                }

                HStack(spacing: 4) {
                    Image(systemName: "dollarsign.circle")
                    Text(String(format: "$%.3f", costCents / 100))
                }
            }
            .font(.caption)
            .foregroundColor(.secondary)

            Spacer()
        }
    }

    private var formattedTime: String {
        if processingTimeMs < 1000 {
            return "\(processingTimeMs)ms"
        } else {
            return String(format: "%.1fs", Double(processingTimeMs) / 1000)
        }
    }
}

// MARK: - Section Header

struct SectionHeader: View {
    let title: String
    let icon: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundColor(.green)

            Text(title)
                .font(.headline)
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        DiagnosisResultView(
            diagnosis: DiagnoseResponse(
                speciesGuess: SpeciesGuess(name: "Monstera deliciosa", confidence: 0.92),
                issues: [
                    Issue(
                        name: "Overwatering",
                        confidence: 0.85,
                        severity: "moderate",
                        visualEvidence: ["Yellow lower leaves", "Mushy stem base"]
                    ),
                    Issue(
                        name: "Low light",
                        confidence: 0.60,
                        severity: "mild",
                        visualEvidence: ["Smaller new leaves", "Leggy growth"]
                    )
                ],
                explanation: "Your Monstera is showing signs of overwatering, which is causing the lower leaves to turn yellow. The soil is staying too wet for too long, which can lead to root rot if not addressed.",
                immediateActions: [
                    "Stop watering immediately and let soil dry out completely",
                    "Move to a brighter location with indirect light",
                    "Check roots for signs of rot (black, mushy)"
                ],
                recoveryPlan: RecoveryPlan(
                    durationWeeks: 3,
                    prognosis: "good",
                    weeks: [
                        WeekPlan(week: 1, focus: "Dry out", tasks: [
                            "Do not water at all this week",
                            "Ensure pot has drainage holes",
                            "Remove any standing water from saucer"
                        ]),
                        WeekPlan(week: 2, focus: "Monitor", tasks: [
                            "Check soil moisture daily with finger test",
                            "Only water if top 2 inches are completely dry",
                            "Watch for new growth as sign of recovery"
                        ]),
                        WeekPlan(week: 3, focus: "Establish routine", tasks: [
                            "Water only when top 2 inches are dry",
                            "Consider repotting if roots were damaged"
                        ])
                    ]
                ),
                confidenceNotes: "Photo quality is good. Diagnosis confidence is high for overwatering, moderate for light issues.",
                processingTimeMs: 3420,
                modelCostCents: 2.34,
                modelUsed: "claude-sonnet-4-5-20250929"
            ),
            image: nil,
            onDiagnoseAnother: {}
        )
    }
}
