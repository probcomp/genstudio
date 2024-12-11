import { createProgram } from './webgl-utils';
import { pickingShaders } from './shaders';

export class PickingSystem {
    private gl: WebGL2RenderingContext;
    private program: WebGLProgram | null;
    private framebuffer: WebGLFramebuffer | null;
    private texture: WebGLTexture | null;
    private radius: number;
    private width: number;
    private height: number;
    private canvas: HTMLCanvasElement;
    private vao: WebGLVertexArrayObject | null = null;

    constructor(gl: WebGL2RenderingContext, pickingRadius: number) {
        this.gl = gl;
        this.canvas = gl.canvas as HTMLCanvasElement;
        this.radius = pickingRadius;
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Create picking program
        this.program = createProgram(gl, pickingShaders.vertex, pickingShaders.fragment);
        if (!this.program) {
            throw new Error('Failed to create picking program');
        }

        // Create framebuffer and texture
        this.framebuffer = gl.createFramebuffer();
        this.texture = gl.createTexture();

        if (!this.framebuffer || !this.texture) {
            throw new Error('Failed to create picking framebuffer');
        }

        // Initialize texture
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA,
            this.width, this.height, 0,
            gl.RGBA, gl.UNSIGNED_BYTE, null
        );

        // Attach texture to framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D, this.texture, 0
        );
    }

    setVAO(vao: WebGLVertexArrayObject) {
        this.vao = vao;
    }

    pick(x: number, y: number): number | null {
        const gl = this.gl;
        if (!this.program || !this.framebuffer || !this.vao) return null;

        // Bind framebuffer and set viewport
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        gl.viewport(0, 0, this.width, this.height);

        // Clear the framebuffer
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // Read pixels in the picking radius
        const pixelX = Math.floor(x * this.width / this.canvas.clientWidth);
        const pixelY = Math.floor(this.height - y * this.height / this.canvas.clientHeight);

        const pixels = new Uint8Array(4 * this.radius * this.radius);
        gl.readPixels(
            pixelX - this.radius/2,
            pixelY - this.radius/2,
            this.radius,
            this.radius,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            pixels
        );

        // Find closest point
        let closestPoint: number | null = null;
        let minDist = Infinity;

        for (let i = 0; i < pixels.length; i += 4) {
            if (pixels[i+3] === 0) continue; // Skip empty pixels

            const pointId = pixels[i] + pixels[i+1] * 256 + pixels[i+2] * 256 * 256;
            const dx = (i/4) % this.radius - this.radius/2;
            const dy = Math.floor((i/4) / this.radius) - this.radius/2;
            const dist = dx*dx + dy*dy;

            if (dist < minDist) {
                minDist = dist;
                closestPoint = pointId;
            }
        }

        // Restore default framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        gl.bindVertexArray(this.vao);

        return closestPoint;
    }

    cleanup(): void {
        const gl = this.gl;

        if (this.framebuffer) {
            gl.deleteFramebuffer(this.framebuffer);
            this.framebuffer = null;
        }

        if (this.texture) {
            gl.deleteTexture(this.texture);
            this.texture = null;
        }

        if (this.program) {
            gl.deleteProgram(this.program);
            this.program = null;
        }
    }
}
