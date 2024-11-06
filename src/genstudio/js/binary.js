/**
 * Reshapes a flat array into a nested array structure based on the provided dimensions.
 * The input array can be either a TypedArray (like Float32Array) or regular JavaScript Array.
 * The leaf arrays (deepest level) maintain the original array type, while the nested structure
 * uses regular JavaScript arrays.
 *
 * @param {TypedArray|Array} flat - The flat array to reshape
 * @param {number[]} dims - Array of dimensions specifying the desired shape
 * @param {number} [offset=0] - Starting offset into the flat array (used internally for recursion)
 * @returns {Array} A nested array matching the specified dimensions, with leaves maintaining the original type
 *
 * @example
 * // With regular Array
 * reshapeArray([1,2,3,4], [2,2])
 * // Returns: [[1,2], [3,4]]
 *
 * @example
 * // With TypedArray
 * const data = new Float32Array([1,2,3,4])
 * reshapeArray(data, [2,2])
 * // Returns nested arrays containing Float32Array slices:
 * // [Float32Array[1,2], Float32Array[3,4]]
 */
export function reshapeArray(flat, dims, offset = 0) {
  const [dim, ...restDims] = dims;

  if (restDims.length === 0) {
    const start = offset;
    const end = offset + dim;
    return flat.slice(start, end);
  }

  const stride = restDims.reduce((a, b) => a * b, 1);
  return Array.from({ length: dim }, (_, i) =>
    reshapeArray(flat, restDims, offset + i * stride)
  );
}

/**
 * Evaluates an ndarray node by converting the DataView buffer into a typed array
 * and optionally reshaping it into a multidimensional array.
 *
 * @param {Object} node - The ndarray node to evaluate
 * @param {DataView} node.data - The raw binary data as a DataView
 * @param {string} node.dtype - The data type (e.g. 'float32', 'int32')
 * @param {number[]} node.shape - The shape of the array
 * @returns {TypedArray|Array} For 1D arrays, returns a typed array. For higher dimensions,
 *                            returns a nested JavaScript array matching the shape.
 */
export function evaluateNdarray(node) {
  const { data, dtype, shape } = node;
  const dtypeMap = {
    'float32': Float32Array,
    'float64': Float64Array,
    'int8': Int8Array,
    'int16': Int16Array,
    'int32': Int32Array,
    'uint8': Uint8Array,
    'uint16': Uint16Array,
    'uint32': Uint32Array,
  };
  const ArrayConstructor = dtypeMap[dtype] || Float64Array;

  // Create typed array directly from the DataView's buffer
  const flatArray = new ArrayConstructor(
    data.buffer,
    data.byteOffset,
    data.byteLength / ArrayConstructor.BYTES_PER_ELEMENT
  );
  // If 1D, return the typed array directly
  if (shape.length <= 1) {
    return flatArray;
  }

  return reshapeArray(flatArray, shape);
}

/**
 * Estimates the size of a JSON string in bytes and returns a human readable string.
 *
 * @param {string} jsonString - The JSON string to measure
 * @returns {string} Human readable size (e.g. "1.23 KB" or "4.56 MB")
 */
export function estimateJSONSize(jsonString) {
  if (!jsonString) return '0 B';

  // Use TextEncoder to get accurate byte size for UTF-8 encoded string
  const encoder = new TextEncoder();
  const bytes = encoder.encode(jsonString).length;

  // Convert bytes to KB or MB
  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  } else {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
}
