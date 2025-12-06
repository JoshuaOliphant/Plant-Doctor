//
//  Models.swift
//  PlantDocPoC
//
//  Data models for API communication
//

import Foundation

// MARK: - Request Models

struct DiagnoseRequest: Codable {
    let image: String  // Base64 encoded
    let context: String?
}

// MARK: - Response Models

struct DiagnoseResponse: Codable {
    let speciesGuess: SpeciesGuess
    let issues: [Issue]
    let explanation: String
    let immediateActions: [String]
    let recoveryPlan: RecoveryPlan
    let confidenceNotes: String
    let processingTimeMs: Int
    let modelCostCents: Double

    enum CodingKeys: String, CodingKey {
        case speciesGuess = "species_guess"
        case issues
        case explanation
        case immediateActions = "immediate_actions"
        case recoveryPlan = "recovery_plan"
        case confidenceNotes = "confidence_notes"
        case processingTimeMs = "processing_time_ms"
        case modelCostCents = "model_cost_cents"
    }
}

struct SpeciesGuess: Codable {
    let name: String
    let confidence: Double
}

struct Issue: Codable, Identifiable {
    var id: String { name }
    let name: String
    let confidence: Double
    let severity: String
    let visualEvidence: [String]

    enum CodingKeys: String, CodingKey {
        case name
        case confidence
        case severity
        case visualEvidence = "visual_evidence"
    }

    var severityColor: String {
        switch severity.lowercased() {
        case "mild": return "green"
        case "moderate": return "orange"
        case "severe": return "red"
        default: return "gray"
        }
    }
}

struct RecoveryPlan: Codable {
    let durationWeeks: Int
    let prognosis: String
    let weeks: [WeekPlan]

    enum CodingKeys: String, CodingKey {
        case durationWeeks = "duration_weeks"
        case prognosis
        case weeks
    }

    var prognosisEmoji: String {
        switch prognosis.lowercased() {
        case "good": return "+"
        case "fair": return "~"
        case "guarded": return "!"
        default: return "?"
        }
    }
}

struct WeekPlan: Codable, Identifiable {
    var id: Int { week }
    let week: Int
    let focus: String
    let tasks: [String]
}

// MARK: - Error Models

struct APIError: Codable {
    let detail: String
}
