const fs = require('fs');
const path = require('path');

function toHex(val) {
    const isNeg = val < 0;
    const absVal = BigInt(Math.floor(Math.abs(val)));
    return (isNeg ? '-' : '') + '0x' + absVal.toString(16);
}

function rand(limit = 0x1fff) {
    return Math.floor(Math.random() * (limit * 2)) - limit;
}

const convWeights = [];
for (let ky = 0; ky < 3; ky++) {
    const row = [];
    for (let kx = 0; kx < 3; kx++) {
        const channelsIn = [];
        for (let c_in = 0; c_in < 3; c_in++) {
            const channelsOut = [];
            for (let c_out = 0; c_out < 2; c_out++) {
                channelsOut.push(toHex(rand()));
            }
            channelsIn.push(`[${channelsOut.join(',')}]`);
        }
        row.push(`[${channelsIn.join(',')}]`);
    }
    convWeights.push(`[${row.join(',')}]`);
}

const convBias = [toHex(rand(0x10000)), toHex(rand(0x10000))];

const denseWeights = [];
for (let out = 0; out < 16; out++) {
    const row = [];
    for (let i = 0; i < 392; i++) {
        row.push(toHex(rand()));
    }
    denseWeights.push(`[${row.join(',')}]`);
}

const denseBias = Array(16).fill(0).map(() => toHex(rand(0x40000)));

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
	var embedding_layer_W[16][392];
	embedding_layer_W = [${denseWeights.join(',')}];
	return embedding_layer_W;
}

function get_embedding_layer_b(){
	var embedding_layer_b[16];
	embedding_layer_b = [${denseBias.join(',')}];
	return embedding_layer_b;
}
`;

fs.writeFileSync(path.join(__dirname, 'compiled_circuit/model_circom/weights_v30.circom'), content);
console.log("Generated optimized 30x30 weights successfully!");
