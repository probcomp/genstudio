import { describe, expect, it } from 'vitest';
import { inferDtype, evaluateNdarray, estimateJSONSize } from '../../src/genstudio/js/binary';

describe('binary.js', () => {
  describe('inferDtype', () => {
    it('should infer dtype from TypedArray', () => {
      expect(inferDtype(new Float32Array([1, 2, 3]))).toBe('float32');
      expect(inferDtype(new Int16Array([1, 2, 3]))).toBe('int16');
      expect(inferDtype(new Uint8Array([1, 2, 3]))).toBe('uint8');
    });

    it('should throw error for non-TypedArray input', () => {
      expect(() => inferDtype([1, 2, 3])).toThrow('Value must be a TypedArray');
      expect(() => inferDtype('abc')).toThrow('Value must be a TypedArray');
    });
  });

  describe('evaluateNdarray', () => {
    it('should evaluate 1D float32 array', () => {
      const data = new Float32Array([1, 2, 3, 4]).buffer;
      const node = {
        data: new DataView(data),
        dtype: 'float32',
        shape: [4]
      };
      const result = evaluateNdarray(node);
      expect(result).toBeInstanceOf(Float32Array);
      expect(Array.from(result)).toEqual([1, 2, 3, 4]);
    });

    it('should evaluate 1D uint8 array', () => {
      const data = new Uint8Array([1, 2, 3, 4]).buffer;
      const node = {
        data: new DataView(data),
        dtype: 'uint8',
        shape: [4]
      };
      const result = evaluateNdarray(node);
      expect(result).toBeInstanceOf(Uint8Array);
      expect(Array.from(result)).toEqual([1, 2, 3, 4]);
    });

    it('should handle unknown dtype by defaulting to Float64Array', () => {
      const data = new Float64Array([1, 2]).buffer;
      const node = {
        data: new DataView(data),
        dtype: 'unknown',
        shape: [2]
      };
      const result = evaluateNdarray(node);
      expect(result).toBeInstanceOf(Float64Array);
    });
  });

  describe('estimateJSONSize', () => {
    it('should return "0 B" for empty input', () => {
      expect(estimateJSONSize('')).toBe('0 B');
      expect(estimateJSONSize(null)).toBe('0 B');
    });

    it('should estimate bytes correctly', () => {
      expect(estimateJSONSize('abc')).toBe('3 B');
      expect(estimateJSONSize('🌟')).toBe('4 B'); // UTF-8 encoded emoji
    });

    it('should format KB correctly', () => {
      const kilobyteString = 'x'.repeat(1024);
      expect(estimateJSONSize(kilobyteString)).toBe('1.00 KB');
    });

    it('should format MB correctly', () => {
      const megabyteString = 'x'.repeat(1024 * 1024);
      expect(estimateJSONSize(megabyteString)).toBe('1.00 MB');
    });
  });
});
