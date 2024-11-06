import { describe, expect, it } from 'vitest';
import { reshapeArray, evaluateNdarray, estimateJSONSize } from '../../src/genstudio/js/binary';

describe('binary.js', () => {
  describe('reshapeArray', () => {
    it('should reshape 1D array to 2D', () => {
      const flat = [1, 2, 3, 4];
      const result = reshapeArray(flat, [2, 2]);
      expect(result).toEqual([[1, 2], [3, 4]]);
    });

    it('should reshape TypedArray to 2D while preserving type', () => {
      const flat = new Float32Array([1, 2, 3, 4]);
      const result = reshapeArray(flat, [2, 2]);
      expect(result[0]).toBeInstanceOf(Float32Array);
      expect(result[1]).toBeInstanceOf(Float32Array);
      expect(Array.from(result[0])).toEqual([1, 2]);
      expect(Array.from(result[1])).toEqual([3, 4]);
    });

    it('should handle 3D reshaping', () => {
      const flat = [1, 2, 3, 4, 5, 6, 7, 8];
      const result = reshapeArray(flat, [2, 2, 2]);
      expect(result).toEqual([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
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

    it('should evaluate 2D uint8 array', () => {
      const data = new Uint8Array([1, 2, 3, 4]).buffer;
      const node = {
        data: new DataView(data),
        dtype: 'uint8',
        shape: [2, 2]
      };
      const result = evaluateNdarray(node);
      expect(Array.isArray(result)).toBe(true);
      expect(result[0]).toBeInstanceOf(Uint8Array);
      expect(Array.from(result[0])).toEqual([1, 2]);
      expect(Array.from(result[1])).toEqual([3, 4]);
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
