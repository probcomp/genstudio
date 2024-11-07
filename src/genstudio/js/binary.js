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

/**
 * Infers the numpy dtype for a JavaScript array by examining its contents
 *
 * @param {Array} arr - JavaScript array to analyze
 * @returns {string} Numpy dtype string (e.g. 'float32', 'int32', etc)
 */
export function inferDtype(value) {

  if (!(value instanceof ArrayBuffer || ArrayBuffer.isView(value))) {
    throw new Error('Value must be a TypedArray');
  }

  return value.constructor.name.toLowerCase().replace('array', '');
}

/**
 * Evaluates an ndarray node by converting the DataView buffer into a typed array.
 * Only handles 1D arrays - higher dimensions will be rejected during serialization.
 *
 * @param {Object} node - The ndarray node to evaluate
 * @param {DataView} node.data - The raw binary data as a DataView
 * @param {string} node.dtype - The data type (e.g. 'float32', 'int32')
 * @param {number[]} node.shape - The shape of the array (must be 1D)
 * @returns {TypedArray} A typed array containing the data
 */
export function evaluateNdarray(node) {
  const { data, dtype, shape } = node;
  const ArrayConstructor = dtypeMap[dtype] || Float64Array;

  // Create typed array directly from the DataView's buffer
  return new ArrayConstructor(
    data.buffer,
    data.byteOffset,
    data.byteLength / ArrayConstructor.BYTES_PER_ELEMENT
  );
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
