package com.llmhub.llmhub.agent

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class TermuxOutputStreamTest {
    @Test fun streamsStdoutAndStderrBeforeProcessFinishesWithoutWaitingForNewlines() {
        TermuxOutputStream().use { stream ->
            val executor = Executors.newSingleThreadExecutor()
            val early = CountDownLatch(1)
            val text = StringBuffer()
            val collector = executor.submit {
                stream.collect { chunk -> text.append(chunk); if (text.contains("early")) early.countDown() }
            }
            val process = ProcessBuilder("bash", "-c", stream.commandScript("printf early; sleep 1; printf 'late\\n' >&2"))
                .redirectErrorStream(true).start()
            try {
                assertTrue("Output must arrive during execution", early.await(3, TimeUnit.SECONDS))
                assertTrue("Command must still be running when first bytes arrive", process.isAlive)
                assertTrue(stream.processGroup!! > 1)
                assertEquals(0, process.waitFor())
                collector.get(3, TimeUnit.SECONDS)
                assertEquals("earlylate\n", text.toString())
            } finally { process.destroyForcibly(); stream.close(); executor.shutdownNow() }
        }
    }
    @Test fun preservesNonzeroCommandExitAndItsOutput() {
        TermuxOutputStream().use { stream ->
            val executor = Executors.newSingleThreadExecutor()
            val text = StringBuffer()
            val collector = executor.submit { stream.collect { text.append(it) } }
            val process = ProcessBuilder("bash", "-c", stream.commandScript("printf 'broken\\n' >&2; exit 7"))
                .redirectErrorStream(true).start()
            try {
                assertEquals(7, process.waitFor())
                collector.get(3, TimeUnit.SECONDS)
                assertEquals("broken\n", text.toString())
            } finally { process.destroyForcibly(); stream.close(); executor.shutdownNow() }
        }
    }
    @Test fun bufferCapsOutputAndRemovesTerminalControlSequences() {
        val output = TerminalOutputBuffer(20)
        assertEquals("red\n", output.append("\u001b[31mred\u001b[0m\r\n"))
        assertEquals("x".repeat(20), output.append("x".repeat(100)))
    }
    @Test fun cancellingTheWorkerGroupStopsItsDescendantsAndClosesOutput() {
        TermuxOutputStream().use { stream ->
            val executor = Executors.newSingleThreadExecutor()
            val early = CountDownLatch(1)
            val collector = executor.submit { stream.collect { if (it.contains("ready")) early.countDown() } }
            val process = ProcessBuilder("bash", "-c", stream.commandScript("printf ready; sleep 30"))
                .redirectErrorStream(true).start()
            try {
                assertTrue(early.await(3, TimeUnit.SECONDS))
                val group = stream.processGroup!!
                val ps = ProcessBuilder("ps", "-o", "pgid=", "-p", group.toString()).start()
                assertEquals(group.toString(), ps.inputStream.bufferedReader().readText().trim())
                assertEquals(0, ProcessBuilder("bash", "-c", "kill -TERM -- -$group").start().waitFor())
                assertTrue("Worker and wrapper must exit after cancellation", process.waitFor(3, TimeUnit.SECONDS))
                collector.get(3, TimeUnit.SECONDS)
            } finally { process.destroyForcibly(); stream.close(); executor.shutdownNow() }
        }
    }

}
