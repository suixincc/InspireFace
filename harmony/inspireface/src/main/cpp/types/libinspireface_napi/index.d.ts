export interface NativeSessionHandle {
  readonly __inspireFaceSessionBrand: string;
}

export interface NativeImageStreamHandle {
  readonly __inspireFaceImageStreamBrand: string;
}

export interface NativeVersion {
  major: number;
  minor: number;
  patch: number;
}

export interface NativeSessionOptions {
  featureMask?: number;
  detectMode?: number;
  maxFaces?: number;
  detectPixelLevel?: number;
  trackFps?: number;
}

export interface NativeSessionRuntimeOptions {
  previewSize?: number;
  minimumFaceSize?: number;
  smoothCacheFrames?: number;
  detectInterval?: number;
  landmarkAugmentation?: number;
  detectThreshold?: number;
  smoothRatio?: number;
  lightTrackThreshold?: number;
  trackLostRecovery?: boolean;
}

export interface NativeRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NativeTrackedFace {
  rect: NativeRect;
  trackId: number;
  trackCount: number;
  confidence: number;
  roll: number;
  yaw: number;
  pitch: number;
  token: Uint8Array;
}

export interface NativeTrackResult {
  detectedNum: number;
  faces: NativeTrackedFace[];
}

interface InspireFaceNativeModule {
  launch(resourcePath: string): void;
  reload(resourcePath: string): void;
  terminate(): void;
  isLaunched(): boolean;
  getVersion(): NativeVersion;
  getCapiLevel(): number;
  createSession(options: NativeSessionOptions): NativeSessionHandle;
  releaseSession(handle: NativeSessionHandle): void;
  configureSession(handle: NativeSessionHandle, options: NativeSessionRuntimeOptions): void;
  clearTracking(handle: NativeSessionHandle): void;
  createImageStream(data: Uint8Array, width: number, height: number, format: number,
    rotation: number): NativeImageStreamHandle;
  releaseImageStream(handle: NativeImageStreamHandle): void;
  track(session: NativeSessionHandle, image: NativeImageStreamHandle): NativeTrackResult;
  getDenseLandmarks(token: Uint8Array): Float32Array;
  getFiveKeyPoints(token: Uint8Array): Float32Array;
  extractFeature(session: NativeSessionHandle, image: NativeImageStreamHandle,
    token: Uint8Array): Float32Array;
  getFeatureLength(): number;
  compareFeatures(first: Float32Array, second: Float32Array): number;
  getRecommendedThreshold(): number;
  similarityToPercentage(similarity: number): number;
  setLogLevel(level: number): void;
  disableLog(): void;
}

declare const inspireFaceNative: InspireFaceNativeModule;
export default inspireFaceNative;
