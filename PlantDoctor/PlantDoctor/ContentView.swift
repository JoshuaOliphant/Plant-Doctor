//
//  ContentView.swift
//  PlantDoctor
//
//  Main flow coordinator: Home -> Camera -> Loading -> Result
//

import SwiftUI
import PhotosUI
internal import Combine

enum AppState {
    case home
    case camera
    case classifying
    case classification(ClassifyResponse)
    case diagnosing
    case result(DiagnoseResponse)
    case error(String)
}

struct ContentView: View {
    @State private var appState: AppState = .home
    @State private var selectedImage: UIImage?
    @State private var userContext: String = ""
    @State private var showingImagePicker = false
    @State private var showingCamera = false

    // Configure your backend URL here
    private let apiClient = APIClient(baseURL: "http://localhost:8001")

    var body: some View {
        NavigationStack {
            Group {
                switch appState {
                case .home:
                    HomeView(
                        onTakePhoto: { showingCamera = true },
                        onChoosePhoto: { showingImagePicker = true }
                    )

                case .camera:
                    // Camera is presented as a sheet
                    HomeView(
                        onTakePhoto: { showingCamera = true },
                        onChoosePhoto: { showingImagePicker = true }
                    )

                case .classifying:
                    LoadingView(message: "Classifying your plant")

                case .classification(let classification):
                    ClassificationResultView(
                        classification: classification,
                        image: selectedImage,
                        onGetDiagnosis: { submitDiagnosis() },
                        onDiagnoseAnother: { resetToHome() }
                    )

                case .diagnosing:
                    LoadingView(message: "Getting detailed diagnosis")

                case .result(let diagnosis):
                    DiagnosisResultView(
                        diagnosis: diagnosis,
                        image: selectedImage,
                        onDiagnoseAnother: { resetToHome() }
                    )

                case .error(let message):
                    ErrorView(
                        message: message,
                        onRetry: { submitClassification() },
                        onBack: { resetToHome() }
                    )
                }
            }
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(image: $selectedImage, sourceType: .photoLibrary)
            }
            .sheet(isPresented: $showingCamera) {
                ImagePicker(image: $selectedImage, sourceType: .camera)
            }
            .onChange(of: selectedImage) { _, newImage in
                if newImage != nil {
                    showContextInput()
                }
            }
        }
    }

    private func showContextInput() {
        // Show context input sheet, then submit
        appState = .home
        // For PoC simplicity, we'll go straight to classification
        // In a real app, you'd show a context input modal here
        submitClassification()
    }

    private func submitClassification() {
        guard let image = selectedImage else { return }

        appState = .classifying

        Task {
            do {
                let result = try await apiClient.classify(image: image)
                await MainActor.run {
                    appState = .classification(result)
                }
            } catch {
                await MainActor.run {
                    appState = .error(error.localizedDescription)
                }
            }
        }
    }

    private func submitDiagnosis() {
        guard let image = selectedImage else { return }

        appState = .diagnosing

        Task {
            do {
                let context = userContext.isEmpty ? nil : userContext
                let result = try await apiClient.diagnose(image: image, context: context)
                await MainActor.run {
                    appState = .result(result)
                }
            } catch {
                await MainActor.run {
                    appState = .error(error.localizedDescription)
                }
            }
        }
    }

    private func resetToHome() {
        selectedImage = nil
        userContext = ""
        appState = .home
    }
}

// MARK: - Home View

struct HomeView: View {
    let onTakePhoto: () -> Void
    let onChoosePhoto: () -> Void

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            // App Icon/Logo area
            VStack(spacing: 16) {
                Image(systemName: "leaf.fill")
                    .font(.system(size: 80))
                    .foregroundColor(.green)

                Text("PlantDoc")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("AI-powered plant diagnosis")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Explainer
            VStack(alignment: .leading, spacing: 12) {
                ExplainerRow(
                    icon: "camera.fill",
                    text: "Take a photo of your plant"
                )
                ExplainerRow(
                    icon: "cpu",
                    text: "AI analyzes visible symptoms"
                )
                ExplainerRow(
                    icon: "heart.text.square.fill",
                    text: "Get diagnosis & recovery plan"
                )
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(16)
            .padding(.horizontal)

            Spacer()

            // Action buttons
            VStack(spacing: 16) {
                Button(action: onTakePhoto) {
                    Label("Take Photo", systemImage: "camera.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }

                Button(action: onChoosePhoto) {
                    Label("Choose from Library", systemImage: "photo.on.rectangle")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.systemGray5))
                        .foregroundColor(.primary)
                        .cornerRadius(12)
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 32)
        }
        .navigationBarHidden(true)
    }
}

struct ExplainerRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.green)
                .frame(width: 32)

            Text(text)
                .font(.subheadline)
        }
    }
}

// MARK: - Loading View

struct LoadingView: View {
    let message: String
    @State private var dots = ""
    let timer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    init(message: String = "Analyzing your plant") {
        self.message = message
    }

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            ProgressView()
                .scaleEffect(1.5)

            Text("\(message)\(dots)")
                .font(.title2)
                .fontWeight(.medium)

            Text("This may take a few seconds")
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()
        }
        .onReceive(timer) { _ in
            if dots.count >= 3 {
                dots = ""
            } else {
                dots += "."
            }
        }
    }
}

// MARK: - Error View

struct ErrorView: View {
    let message: String
    let onRetry: () -> Void
    let onBack: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.orange)

            Text("Diagnosis Failed")
                .font(.title)
                .fontWeight(.bold)

            Text(message)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Spacer()

            VStack(spacing: 12) {
                Button(action: onRetry) {
                    Text("Try Again")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }

                Button(action: onBack) {
                    Text("Start Over")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.systemGray5))
                        .foregroundColor(.primary)
                        .cornerRadius(12)
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 32)
        }
    }
}

// MARK: - Classification Result View

struct ClassificationResultView: View {
    let classification: ClassifyResponse
    let image: UIImage?
    let onGetDiagnosis: () -> Void
    let onDiagnoseAnother: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header with plant image
                if let image = image {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(height: 200)
                        .clipped()
                        .cornerRadius(12)
                }

                // Classification results
                VStack(spacing: 16) {
                    // Species
                    VStack(spacing: 8) {
                        HStack {
                            Image(systemName: "leaf.fill")
                                .foregroundColor(.green)
                            Text("Species")
                                .font(.headline)
                        }

                        Text(classification.species)
                            .font(.title2)
                            .fontWeight(.bold)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Disease/Condition
                    VStack(spacing: 8) {
                        HStack {
                            Image(systemName: "cross.case.fill")
                                .foregroundColor(severityColor)
                            Text("Condition")
                                .font(.headline)
                        }

                        Text(classification.disease)
                            .font(.title3)
                            .fontWeight(.semibold)

                        HStack {
                            SeverityBadge(severity: classification.severity)

                            Spacer()

                            HStack(spacing: 4) {
                                Text("Confidence:")
                                    .foregroundColor(.secondary)
                                Text("\(Int(classification.confidence * 100))%")
                                    .fontWeight(.medium)
                                    .foregroundColor(confidenceColor)
                            }
                            .font(.subheadline)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(severityColor.opacity(0.1))
                    .cornerRadius(12)
                }

                // Health status message
                if !classification.needsDiagnosis {
                    HStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(.green)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Your plant looks healthy!")
                                .font(.headline)
                            Text("No immediate action needed")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(12)
                }

                // Full diagnosis button
                if classification.needsDiagnosis {
                    VStack(spacing: 12) {
                        HStack(spacing: 12) {
                            Image(systemName: "info.circle.fill")
                                .foregroundColor(.orange)

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Issues detected")
                                    .font(.headline)
                                Text("Get a detailed diagnosis and recovery plan")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                        }

                        Button(action: onGetDiagnosis) {
                            Label("Get Full Diagnosis", systemImage: "stethoscope")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(12)
                        }
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(12)
                }

                // Stats footer
                HStack {
                    Spacer()

                    HStack(spacing: 16) {
                        HStack(spacing: 4) {
                            Image(systemName: "clock")
                            Text(formattedTime)
                        }

                        HStack(spacing: 4) {
                            Image(systemName: "dollarsign.circle")
                            Text(String(format: "$%.3f", classification.modelCostCents / 100))
                        }

                        HStack(spacing: 4) {
                            Image(systemName: "cpu")
                            Text("Haiku")
                        }
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)

                    Spacer()
                }

                // Diagnose another button
                Button(action: onDiagnoseAnother) {
                    Label("Diagnose Another Plant", systemImage: "camera.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.systemGray5))
                        .foregroundColor(.primary)
                        .cornerRadius(12)
                }
            }
            .padding()
        }
        .navigationTitle("Classification")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var severityColor: Color {
        switch classification.severity.lowercased() {
        case "mild": return .green
        case "moderate": return .orange
        case "severe": return .red
        default: return .gray
        }
    }

    private var confidenceColor: Color {
        switch classification.confidence {
        case 0.8...1.0: return .green
        case 0.5..<0.8: return .orange
        default: return .red
        }
    }

    private var formattedTime: String {
        if classification.processingTimeMs < 1000 {
            return "\(classification.processingTimeMs)ms"
        } else {
            return String(format: "%.1fs", Double(classification.processingTimeMs) / 1000)
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
