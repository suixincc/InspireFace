import native, { NativeImageStreamHandle, NativeSessionHandle } from 'libinspireface_napi.so';

export enum Feature {
  NONE = 0x00000000,
  FACE_RECOGNITION = 0x00000002,
  LIVENESS = 0x00000004,
  IR_LIVENESS = 0x00000008,
  MASK_DETECT = 0x00000010,
  FACE_ATTRIBUTE = 0x00000020,
  QUALITY = 0x00000080,
  INTERACTION = 0x00000100,
  FACE_POSE = 0x00000200,
  FACE_EMOTION = 0x00000400
}

export enum DetectMode {
  ALWAYS_DETECT = 0,
  LIGHT_TRACK = 1,
  TRACK_BY_DETECTION = 2
}

export enum ImageFormat {
  RGB = 0,
  BGR = 1,
  RGBA = 2,
  BGRA = 3,
  YUV_NV12 = 4,
  YUV_NV21 = 5,
  I420 = 6,
  GRAY = 7
}

export enum Rotation {
  DEGREE_0 = 0,
  DEGREE_90 = 1,
  DEGREE_180 = 2,
  DEGREE_270 = 3
}

export enum LogLevel {
  NONE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5
}

export interface Version {
  major: number;
  minor: number;
  patch: number;
}

export interface SessionOptions {
  featureMask?: number;
  detectMode?: DetectMode;
  maxFaces?: number;
  detectPixelLevel?: number;
  trackFps?: number;
}

export interface SessionRuntimeOptions {
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

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TrackedFace {
  rect: Rect;
  trackId: number;
  trackCount: number;
  confidence: number;
  roll: number;
  yaw: number;
  pitch: number;
  token: Uint8Array;
}

export interface FaceTrackResult {
  detectedNum: number;
  faces: TrackedFace[];
}

interface SessionState {
  handle: NativeSessionHandle;
  closed: boolean;
}

interface ImageStreamState {
  handle: NativeImageStreamHandle;
  closed: boolean;
}

const sessionStates: WeakMap<Session, SessionState> = new WeakMap<Session, SessionState>();
const imageStreamStates: WeakMap<ImageStream, ImageStreamState> = new WeakMap<ImageStream, ImageStreamState>();

function requireSessionState(session: Session): SessionState {
  const state: SessionState | undefined = sessionStates.get(session);
  if (state === undefined || state.closed) {
    throw new Error('InspireFace session has already been closed');
  }
  return state;
}

function requireImageStreamState(image: ImageStream): ImageStreamState {
  const state: ImageStreamState | undefined = imageStreamStates.get(image);
  if (state === undefined || state.closed) {
    throw new Error('InspireFace image stream has already been closed');
  }
  return state;
}

/**
 * Owns an immutable native copy of one image. Call close() when processing is complete.
 */
export class ImageStream {
  public constructor(data: Uint8Array, width: number, height: number, format: ImageFormat,
    rotation: Rotation = Rotation.DEGREE_0) {
    const handle: NativeImageStreamHandle = native.createImageStream(data, width, height, format, rotation);
    imageStreamStates.set(this, { handle: handle, closed: false });
  }

  public close(): void {
    const state: ImageStreamState | undefined = imageStreamStates.get(this);
    if (state === undefined || state.closed) {
      return;
    }
    native.releaseImageStream(state.handle);
    state.closed = true;
  }

  public isClosed(): boolean {
    const state: ImageStreamState | undefined = imageStreamStates.get(this);
    return state === undefined || state.closed;
  }
}

/**
 * Owns one face-processing session. Calls on the same session are serialized by the native bridge.
 */
export class Session {
  public constructor(options: SessionOptions = {}) {
    const handle: NativeSessionHandle = native.createSession(options);
    sessionStates.set(this, { handle: handle, closed: false });
  }

  public configure(options: SessionRuntimeOptions): void {
    native.configureSession(requireSessionState(this).handle, options);
  }

  public clearTracking(): void {
    native.clearTracking(requireSessionState(this).handle);
  }

  public track(image: ImageStream): FaceTrackResult {
    const session: SessionState = requireSessionState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    return native.track(session.handle, stream.handle);
  }

  public extractFeature(image: ImageStream, face: TrackedFace): Float32Array {
    const session: SessionState = requireSessionState(this);
    const stream: ImageStreamState = requireImageStreamState(image);
    return native.extractFeature(session.handle, stream.handle, face.token);
  }

  public close(): void {
    const state: SessionState | undefined = sessionStates.get(this);
    if (state === undefined || state.closed) {
      return;
    }
    native.releaseSession(state.handle);
    state.closed = true;
  }

  public isClosed(): boolean {
    const state: SessionState | undefined = sessionStates.get(this);
    return state === undefined || state.closed;
  }
}

export class InspireFace {
  public static launch(resourcePath: string): void {
    native.launch(resourcePath);
  }

  public static reload(resourcePath: string): void {
    native.reload(resourcePath);
  }

  public static terminate(): void {
    native.terminate();
  }

  public static isLaunched(): boolean {
    return native.isLaunched();
  }

  public static getVersion(): Version {
    return native.getVersion();
  }

  public static getCapiLevel(): number {
    return native.getCapiLevel();
  }

  public static createSession(options: SessionOptions = {}): Session {
    return new Session(options);
  }

  public static createImageStream(data: Uint8Array, width: number, height: number, format: ImageFormat,
    rotation: Rotation = Rotation.DEGREE_0): ImageStream {
    return new ImageStream(data, width, height, format, rotation);
  }

  public static getDenseLandmarks(face: TrackedFace): Float32Array {
    return native.getDenseLandmarks(face.token);
  }

  public static getFiveKeyPoints(face: TrackedFace): Float32Array {
    return native.getFiveKeyPoints(face.token);
  }

  public static getFeatureLength(): number {
    return native.getFeatureLength();
  }

  public static compareFeatures(first: Float32Array, second: Float32Array): number {
    return native.compareFeatures(first, second);
  }

  public static getRecommendedThreshold(): number {
    return native.getRecommendedThreshold();
  }

  public static similarityToPercentage(similarity: number): number {
    return native.similarityToPercentage(similarity);
  }

  public static setLogLevel(level: LogLevel): void {
    native.setLogLevel(level);
  }

  public static disableLog(): void {
    native.disableLog();
  }
}
