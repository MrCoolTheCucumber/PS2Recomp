// SPDX-License-Identifier: GPL-3.0-or-later

    .text
    .globl vuNativeSpikeProbeCalleeSaved
    .type vuNativeSpikeProbeCalleeSaved, @function
vuNativeSpikeProbeCalleeSaved:
    pushq %rbx
    pushq %rbp
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq $24, %rsp

    movq %rdi, 0(%rsp)
    movq %rsi, 8(%rsp)
    movq %rdx, 16(%rsp)

    movabsq $0x1122334455667788, %rbx
    movabsq $0x2233445566778899, %rbp
    movabsq $0x33445566778899aa, %r12
    movabsq $0x445566778899aabb, %r13
    movabsq $0x5566778899aabbcc, %r14
    movabsq $0x66778899aabbccdd, %r15

    movq 8(%rsp), %rdi
    call *0(%rsp)

    movq 16(%rsp), %rdx
    movq %rbx, 0(%rdx)
    movq %rbp, 8(%rdx)
    movq %r12, 16(%rdx)
    movq %r13, 24(%rdx)
    movq %r14, 32(%rdx)
    movq %r15, 40(%rdx)

    addq $24, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbp
    popq %rbx
    ret
    .size vuNativeSpikeProbeCalleeSaved, .-vuNativeSpikeProbeCalleeSaved

    .section .note.GNU-stack,"",@progbits
