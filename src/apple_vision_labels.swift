import Foundation
import Vision

guard CommandLine.arguments.count >= 3 else {
    fputs("Usage: apple_vision_labels.swift MANIFEST_JSON IMAGE_ROOT [LIMIT]\n", stderr)
    exit(2)
}

let manifestURL = URL(fileURLWithPath: CommandLine.arguments[1])
let imageRoot = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
let limit = CommandLine.arguments.count > 3 ? Int(CommandLine.arguments[3]) ?? Int.max : Int.max
let raw = try Data(contentsOf: manifestURL)
guard let rows = try JSONSerialization.jsonObject(with: raw) as? [[String: Any]] else {
    fputs("Manifest is not an array\n", stderr)
    exit(2)
}

for (i, row) in rows.prefix(limit).enumerated() {
    guard let id = row["id"] as? Int, let path = row["path"] as? String else {
        fputs("Invalid manifest row \(i)\n", stderr)
        exit(2)
    }
    let url = imageRoot.appendingPathComponent(path)
    let request = VNClassifyImageRequest()
    do {
        try VNImageRequestHandler(url: url).perform([request])
        let labels = (request.results ?? []).prefix(8).map { result -> [String: Any] in
            return ["label": result.identifier, "confidence": Double(result.confidence)]
        }
        let item: [String: Any] = ["id": id, "labels": labels]
        let data = try JSONSerialization.data(withJSONObject: item, options: [.sortedKeys])
        print(String(data: data, encoding: .utf8)!)
    } catch {
        let item: [String: Any] = ["id": id, "labels": [], "error": "\(error)"]
        let data = try JSONSerialization.data(withJSONObject: item, options: [.sortedKeys])
        print(String(data: data, encoding: .utf8)!)
    }
    if (i + 1) % 100 == 0 { fputs("Vision \(i + 1)/\(min(rows.count, limit))\n", stderr) }
}
