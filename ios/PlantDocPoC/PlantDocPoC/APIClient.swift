//
//  APIClient.swift
//  PlantDocPoC
//
//  Network layer for communicating with the PlantDoc backend
//

import Foundation
import UIKit

enum APIClientError: Error, LocalizedError {
    case invalidURL
    case encodingError
    case networkError(Error)
    case serverError(String)
    case decodingError(Error)
    case imageCompressionFailed

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid server URL"
        case .encodingError:
            return "Failed to encode request"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .serverError(let message):
            return message
        case .decodingError(let error):
            return "Failed to parse response: \(error.localizedDescription)"
        case .imageCompressionFailed:
            return "Failed to compress image"
        }
    }
}

actor APIClient {
    // MARK: - Configuration

    // Change this to your deployed backend URL
    // For local testing: "http://localhost:8000"
    // For deployed: "https://your-app.railway.app" or "https://your-app.fly.dev"
    private let baseURL: String

    init(baseURL: String = "http://localhost:8000") {
        self.baseURL = baseURL
    }

    // MARK: - Image Compression

    /// Compress image to target size (default 1MB) while maintaining quality
    private func compressImage(_ image: UIImage, maxSizeKB: Int = 1024) -> Data? {
        var compression: CGFloat = 0.9
        let maxBytes = maxSizeKB * 1024

        guard var imageData = image.jpegData(compressionQuality: compression) else {
            return nil
        }

        // Reduce quality until under target size
        while imageData.count > maxBytes && compression > 0.1 {
            compression -= 0.1
            guard let newData = image.jpegData(compressionQuality: compression) else {
                return nil
            }
            imageData = newData
        }

        // If still too large, resize the image
        if imageData.count > maxBytes {
            let scale = sqrt(Double(maxBytes) / Double(imageData.count))
            let newSize = CGSize(
                width: image.size.width * scale,
                height: image.size.height * scale
            )

            UIGraphicsBeginImageContextWithOptions(newSize, true, 1.0)
            image.draw(in: CGRect(origin: .zero, size: newSize))
            let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()

            if let resized = resizedImage {
                imageData = resized.jpegData(compressionQuality: 0.8) ?? imageData
            }
        }

        return imageData
    }

    // MARK: - API Methods

    /// Diagnose a plant from an image
    func diagnose(image: UIImage, context: String? = nil) async throws -> DiagnoseResponse {
        // Compress image
        guard let imageData = compressImage(image) else {
            throw APIClientError.imageCompressionFailed
        }

        let base64Image = imageData.base64EncodedString()

        // Build request
        guard let url = URL(string: "\(baseURL)/diagnose") else {
            throw APIClientError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60 // Claude can take a while

        let requestBody = DiagnoseRequest(image: base64Image, context: context)

        do {
            request.httpBody = try JSONEncoder().encode(requestBody)
        } catch {
            throw APIClientError.encodingError
        }

        // Make request
        let data: Data
        let response: URLResponse

        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIClientError.networkError(error)
        }

        // Check HTTP status
        if let httpResponse = response as? HTTPURLResponse {
            guard (200...299).contains(httpResponse.statusCode) else {
                // Try to parse error message
                if let apiError = try? JSONDecoder().decode(APIError.self, from: data) {
                    throw APIClientError.serverError(apiError.detail)
                }
                throw APIClientError.serverError("Server error: \(httpResponse.statusCode)")
            }
        }

        // Decode response
        do {
            let decoder = JSONDecoder()
            return try decoder.decode(DiagnoseResponse.self, from: data)
        } catch {
            throw APIClientError.decodingError(error)
        }
    }

    /// Check if the server is reachable
    func healthCheck() async -> Bool {
        guard let url = URL(string: "\(baseURL)/health") else {
            return false
        }

        do {
            let (_, response) = try await URLSession.shared.data(from: url)
            if let httpResponse = response as? HTTPURLResponse {
                return httpResponse.statusCode == 200
            }
            return false
        } catch {
            return false
        }
    }
}
