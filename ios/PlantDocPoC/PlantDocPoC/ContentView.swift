//
//  ContentView.swift
//  PlantDocPoC
//
//  Main flow coordinator: Home -> Camera -> Loading -> Result
//

import SwiftUI
import PhotosUI

enum AppState {
    case home
    case camera
    case loading
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
    private let apiClient = APIClient(baseURL: "http://localhost:8000")

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

                case .loading:
                    LoadingView()

                case .result(let diagnosis):
                    DiagnosisResultView(
                        diagnosis: diagnosis,
                        image: selectedImage,
                        onDiagnoseAnother: { resetToHome() }
                    )

                case .error(let message):
                    ErrorView(
                        message: message,
                        onRetry: { submitDiagnosis() },
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
        // For PoC simplicity, we'll go straight to diagnosis
        // In a real app, you'd show a context input modal here
        submitDiagnosis()
    }

    private func submitDiagnosis() {
        guard let image = selectedImage else { return }

        appState = .loading

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
    @State private var dots = ""
    let timer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            ProgressView()
                .scaleEffect(1.5)

            Text("Analyzing your plant\(dots)")
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

// MARK: - Preview

#Preview {
    ContentView()
}
