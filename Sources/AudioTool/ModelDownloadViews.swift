//
//  ModelDownloadViews.swift
//  ClearVoice
//
//  SwiftUI views for model download management
//

import SwiftUI
import ClearVoiceCore

// MARK: - Model Download Card

/// Card displaying a model with variant picker and download button
@available(macOS 14.0, iOS 17.0, *)
public struct ModelDownloadCard: View {
    public let model: ModelDefinition
    @Bindable public var manager: ModelManager
    
    @State private var selectedVariant: ModelVariant?
    @State private var isExpanded: Bool = false
    
    public init(model: ModelDefinition, manager: ModelManager) {
        self.model = model
        self.manager = manager
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.name)
                        .font(.headline)
                        .lineLimit(2)
                    
                    HStack(spacing: 6) {
                        Image(systemName: model.category.iconName)
                            .foregroundStyle(.secondary)
                        Text(model.category.displayName)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                
                Spacer()
                
                categoryBadge
            }
            
            Text(model.description)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(isExpanded ? nil : 2)
            
            // Variant picker
            if model.variants.count > 1 {
                variantPicker
            }
            
            // Action button
            if let variant = selectedVariant ?? model.variants.first {
                DownloadButton(variant: variant, manager: manager)
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.1), radius: 4, y: 2)
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.2)) {
                isExpanded.toggle()
            }
        }
    }
    
    private var categoryBadge: some View {
        Image(systemName: model.category.iconName)
            .font(.title2)
            .foregroundStyle(.tint)
            .frame(width: 36, height: 36)
            .background(.tint.opacity(0.15))
            .clipShape(Circle())
    }
    
    private var variantPicker: some View {
        HStack {
            Text("Version:")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            Picker("", selection: $selectedVariant) {
                ForEach(model.variants) { variant in
                    HStack {
                        Text(variant.quantization.shortName)
                        Text("•")
                            .foregroundStyle(.secondary)
                        Text(variant.sizeString)
                            .foregroundStyle(.secondary)
                    }
                    .tag(variant as ModelVariant?)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
        }
    }
}

// MARK: - Download Button

/// Progress-aware download/cancel/delete button
@available(macOS 14.0, iOS 17.0, *)
public struct DownloadButton: View {
    public let variant: ModelVariant
    @Bindable public var manager: ModelManager
    
    public init(variant: ModelVariant, manager: ModelManager) {
        self.variant = variant
        self.manager = manager
    }
    
    private var isDownloaded: Bool {
        manager.isDownloaded(variant.id)
    }
    
    private var isDownloading: Bool {
        manager.isDownloading(variant.id)
    }
    
    private var task: DownloadTask? {
        manager.task(for: variant.id)
    }
    
    private var progress: DownloadProgress? {
        manager.progress(for: variant.id)
    }
    
    public var body: some View {
        Group {
            if isDownloaded {
                downloadedView
            } else if isDownloading {
                downloadingView
            } else if let task, case .failed(let message) = task.status {
                failedView(message: message)
            } else {
                downloadView
            }
        }
        .animation(.easeInOut(duration: 0.2), value: isDownloading)
        .animation(.easeInOut(duration: 0.2), value: isDownloaded)
    }
    
    private var downloadView: some View {
        Button {
            manager.download(variant)
        } label: {
            HStack {
                Image(systemName: "arrow.down.circle")
                Text("Download")
                Spacer()
                Text(variant.sizeString)
                    .foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.regular)
    }
    
    private var downloadingView: some View {
        VStack(spacing: 8) {
            HStack {
                ProgressView(value: progress?.fractionCompleted ?? 0)
                    .progressViewStyle(.linear)
                
                Text("\(progress?.percentComplete ?? 0)%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 40, alignment: .trailing)
            }
            
            HStack {
                if let speed = progress?.speedString {
                    Text(speed)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                if let remaining = task?.estimatedTimeRemaining {
                    Text("•")
                        .foregroundStyle(.secondary)
                    Text(formatTime(remaining))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                Button(role: .destructive) {
                    manager.cancel(variant.id)
                } label: {
                    Text("Cancel")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
            }
        }
    }
    
    private var downloadedView: some View {
        HStack {
            Label("Installed", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
            
            Spacer()
            
            Button(role: .destructive) {
                manager.delete(variant)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
        }
    }
    
    private func failedView(message: String) -> some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                Text("Failed")
                    .foregroundStyle(.red)
                Spacer()
            }
            
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            Button {
                manager.retry(variant.id)
            } label: {
                Label("Retry", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
        }
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return "\(Int(seconds))s left"
        } else if seconds < 3600 {
            return "\(Int(seconds / 60))m left"
        } else {
            return "\(Int(seconds / 3600))h left"
        }
    }
}

// MARK: - Downloads View

/// List of active downloads and installed models
@available(macOS 14.0, iOS 17.0, *)
public struct DownloadsView: View {
    @Bindable public var manager: ModelManager
    
    public init(manager: ModelManager) {
        self.manager = manager
    }
    
    public var body: some View {
        List {
            if !manager.activeDownloads.isEmpty {
                Section("Active Downloads") {
                    ForEach(manager.activeDownloads) { task in
                        DownloadTaskRow(task: task, manager: manager)
                    }
                }
            }
            
            if !manager.installedVariants.isEmpty {
                Section {
                    ForEach(manager.installedVariants) { variant in
                        InstalledModelRow(variant: variant, manager: manager)
                    }
                } header: {
                    HStack {
                        Text("Installed Models")
                        Spacer()
                        Text(manager.installedSizeString)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            
            if manager.activeDownloads.isEmpty && manager.installedVariants.isEmpty {
                ContentUnavailableView(
                    "No Models",
                    systemImage: "square.stack.3d.up.slash",
                    description: Text("Download models to get started")
                )
            }
        }
        .navigationTitle("Downloads")
        #if os(iOS)
        .listStyle(.insetGrouped)
        #endif
    }
}

// MARK: - Download Task Row

/// Row displaying an active download task
@available(macOS 14.0, iOS 17.0, *)
public struct DownloadTaskRow: View {
    public let task: DownloadTask
    @Bindable public var manager: ModelManager
    
    public init(task: DownloadTask, manager: ModelManager) {
        self.task = task
        self.manager = manager
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(task.variant.name)
                .font(.headline)
            
            ProgressView(value: task.progress?.fractionCompleted ?? 0)
                .progressViewStyle(.linear)
            
            HStack {
                Text("\(task.progress?.percentComplete ?? 0)%")
                    .font(.caption.monospacedDigit())
                
                if let speed = task.progress?.speedString {
                    Text("•")
                    Text(speed)
                }
                
                Spacer()
                
                Button("Cancel") {
                    manager.cancel(task.id)
                }
                .font(.caption)
            }
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Installed Model Row

/// Row displaying an installed model
@available(macOS 14.0, iOS 17.0, *)
public struct InstalledModelRow: View {
    public let variant: ModelVariant
    @Bindable public var manager: ModelManager
    
    @State private var showDeleteConfirmation = false
    
    public init(variant: ModelVariant, manager: ModelManager) {
        self.variant = variant
        self.manager = manager
    }
    
    public var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(variant.name)
                    .font(.headline)
                
                HStack(spacing: 8) {
                    Text(variant.quantization.shortName)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.secondary.opacity(0.2))
                        .clipShape(Capsule())
                    
                    Text(variant.sizeString)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            Spacer()
            
            Button(role: .destructive) {
                showDeleteConfirmation = true
            } label: {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
        .confirmationDialog(
            "Delete \(variant.name)?",
            isPresented: $showDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                manager.delete(variant)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will remove the model from your device. You can download it again later.")
        }
    }
}

// MARK: - Package Download Card

/// Card for downloading a package of models
@available(macOS 14.0, iOS 17.0, *)
public struct PackageDownloadCard: View {
    public let package: ModelPackage
    @Bindable public var manager: ModelManager
    
    public init(package: ModelPackage, manager: ModelManager) {
        self.package = package
        self.manager = manager
    }
    
    private var isInstalled: Bool {
        manager.isPackageInstalled(package)
    }
    
    private var progress: Double {
        manager.packageProgress(package)
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(package.name)
                        .font(.headline)
                    
                    Text("\(package.modelCount) models • \(package.sizeString)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                Image(systemName: "shippingbox.fill")
                    .font(.title2)
                    .foregroundStyle(.tint)
            }
            
            Text(package.description)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            if isInstalled {
                Label("All Installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else if progress > 0 {
                VStack(spacing: 4) {
                    ProgressView(value: progress)
                    Text("\(Int(progress * 100))% complete")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Button {
                    manager.downloadPackage(package)
                } label: {
                    HStack {
                        Image(systemName: "arrow.down.circle")
                        Text("Download All")
                        Spacer()
                        Text(package.sizeString)
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.1), radius: 4, y: 2)
    }
}

// MARK: - Model Browser View

/// Full browser view for exploring and downloading models
@available(macOS 14.0, iOS 17.0, *)
public struct ModelBrowserView: View {
    @Bindable public var manager: ModelManager
    @State private var selectedCategory: ModelCategory?
    @State private var searchText = ""
    
    public init(manager: ModelManager) {
        self.manager = manager
    }
    
    private var filteredModels: [ModelDefinition] {
        var models = manager.registry.models
        
        if let category = selectedCategory {
            models = models.filter { $0.category == category }
        }
        
        if !searchText.isEmpty {
            models = models.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.description.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        return models
    }
    
    public var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: 16) {
                    // Packages section
                    if selectedCategory == nil && searchText.isEmpty {
                        Section {
                            ForEach(manager.registry.packages) { package in
                                PackageDownloadCard(package: package, manager: manager)
                            }
                        } header: {
                            sectionHeader("Packages")
                        }
                    }
                    
                    // Models section
                    Section {
                        ForEach(filteredModels) { model in
                            ModelDownloadCard(model: model, manager: manager)
                        }
                    } header: {
                        sectionHeader("Models")
                    }
                }
                .padding()
            }
            .navigationTitle("Model Browser")
            .searchable(text: $searchText, prompt: "Search models")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Menu {
                        Button("All Categories") {
                            selectedCategory = nil
                        }
                        Divider()
                        ForEach(ModelCategory.allCases, id: \.self) { category in
                            Button {
                                selectedCategory = category
                            } label: {
                                Label(category.displayName, systemImage: category.iconName)
                            }
                        }
                    } label: {
                        Label("Filter", systemImage: "line.3.horizontal.decrease.circle")
                    }
                }
            }
        }
    }
    
    private func sectionHeader(_ title: String) -> some View {
        HStack {
            Text(title)
                .font(.title2.bold())
            Spacer()
        }
        .padding(.top, 8)
    }
}
