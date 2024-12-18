export function createShader(
    gl: WebGL2RenderingContext,
    type: number,
    source: string
): WebGLShader | null {
    const shader = gl.createShader(type);
    if (!shader) {
        console.error('Failed to create shader');
        return null;
    }

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(
            'Shader compile error:',
            gl.getShaderInfoLog(shader),
            '\nSource:',
            source
        );
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

export function createProgram(
    gl: WebGL2RenderingContext,
    vertexSource: string,
    fragmentSource: string
): WebGLProgram {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource)!;
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource)!;

    const program = gl.createProgram()!;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        throw new Error('Failed to link WebGL program');
    }

    return program;
}
