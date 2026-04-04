const fs = require('fs');
const path = require('path');

/**
 * Generates a 20x20 weights.circom file.
 * Convolution: 3x3 kernel, 3 channels in, 2 channels out.
 * Dense: 162 inputs (9x9x2) to 4 output features.
 */

function toHex(val) {
    const isNeg = val < 0;
    const absVal = Math.abs(val);
    return (isNeg ? '-' : '') + '0x' + absVal.toString(16);
}

function rand() {
    // Generate a reasonable 15-bit quantized weight (-0x3fff to 0x3fff)
    return Math.floor(Math.random() * 0x7fff) - 0x4000;
}

const convWeights = [];
for (let c_in = 0; c_in < 3; c_in++) {
    const channelIn = [];
    for (let ky = 0; ky < 3; ky++) {
        const row = [];
        for (let kx = 0; kx < 3; kx++) {
            const kernels = [];
            for (let c_out = 0; c_out < 2; c_out++) {
                kernels.push(toHex(rand()));
            }
            row.push(`[${kernels.join(',')}]`);
        }
        channelIn.push(`[${row.join(',')}]`);
    }
    convWeights.push(`[${channelIn.join(',')}]`);
}

const convBias = [toHex(rand() * 100), toHex(rand() * 100)];

const denseWeights = [];
for (let out = 0; out < 4; out++) {
    const row = [];
    for (let i = 0; i < 162; i++) {
        row.push(toHex(rand()));
    }
    denseWeights.push(`[${row.join(',')}]`);
}

const denseBias = [toHex(rand() * 1000), toHex(rand() * 1000), toHex(rand() * 1000), toHex(rand() * 1000)];

const content = `pragma circom 2.1.6;

function get_conv2d_weights(){
	var conv2d_weights[3][3][3][2];
	conv2d_weights = [${convWeights.join(',')}];
	return conv2d_weights;
}

function get_conv2d_bias(){
	var conv2d_bias[2];
	conv2d_bias = [${convBias.join(',')}];
	return conv2d_bias;
}

function get_embedding_layer_W(){
	var embedding_layer_W[4][162];
	embedding_layer_W = [${denseWeights.join(',')}];
	return embedding_layer_W;
}

function get_embedding_layer_b(){
	var embedding_layer_b[4];
	embedding_layer_b = [${denseBias.join(',')}];
	return embedding_layer_b;
}
`;

fs.writeFileSync(path.join(__dirname, 'compiled_circuit/model_circom/weights_v20.circom'), content);
console.log("Generated 20x20 weights successfully!");
