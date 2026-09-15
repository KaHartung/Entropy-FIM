import net.sourceforge.plantuml.BlockUml;
import net.sourceforge.plantuml.NewpagedDiagram;
import net.sourceforge.plantuml.SourceStringReader;
import net.sourceforge.plantuml.TitledDiagram;
import net.sourceforge.plantuml.core.Diagram;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Parses PlantUML source with the real PlantUML engine and reports the diagram's
 * validity, its actual DiagramType (core.DiagramType as of PlantUML 1.2026+, which
 * absorbed the fine-grained UML sub-types formerly in the now-removed
 * skin.UmlDiagramType -- MINDMAP/GANTT/JSON and other non-UML families are also
 * core.DiagramType values now, but are only reported here when the underlying
 * Diagram implementation happens to be a TitledDiagram; otherwise the Diagram
 * implementation's class name is reported instead), and how many @start.../@end...
 * blocks the file contains.
 *
 * Two modes:
 * - default: reads one diagram from stdin (until EOF), prints one result line, exits.
 *   Cheap for one-off use, but pays a full JVM boot per invocation.
 * - "--server": stays alive, reading a length-prefixed diagram at a time from stdin
 *   in a loop (so embedded newlines in the diagram source are never ambiguous with
 *   message framing) and printing one result line per diagram, until stdin is closed.
 *   Lets a caller amortize JVM startup across many diagrams instead of paying it
 *   per diagram.
 */
public class DiagramTypeDetector {
    public static void main(String[] args) throws IOException {
        if (args.length > 0 && "--server".equals(args[0])) {
            runServer();
        } else {
            runOnce();
        }
    }

    private static void runOnce() throws IOException {
        String source = readAll(System.in);
        System.out.println(classify(source));
    }

    private static void runServer() throws IOException {
        InputStream in = System.in;
        OutputStream out = System.out;
        while (true) {
            String lengthLine = readLine(in);
            if (lengthLine == null) {
                break; // caller closed stdin: shut down cleanly
            }
            if (lengthLine.isEmpty()) {
                continue;
            }
            int length = Integer.parseInt(lengthLine.trim());
            byte[] payload = in.readNBytes(length);
            if (payload.length != length) {
                throw new EOFException("Expected " + length + " bytes, got " + payload.length);
            }
            String source = new String(payload, StandardCharsets.UTF_8);
            String result = classify(source);
            out.write(result.getBytes(StandardCharsets.UTF_8));
            out.write('\n');
            out.flush();
        }
    }

    private static String classify(String source) {
        try {
            SourceStringReader reader = new SourceStringReader(source);
            List<BlockUml> blocks = reader.getBlocks();

            if (blocks.isEmpty()) {
                return "<result><valid>false</valid><type>NONE</type><blocks>0</blocks></result>";
            }

            // A file can contain more than one @start.../@end... block; PlantUML viewers
            // typically only render the last one, so callers should treat >1 as ambiguous
            // rather than trusting the first block's type.
            Diagram diagram = blocks.get(0).getDiagram();

            // A diagram containing `newpage` directives is still one logical diagram --
            // PlantUML just wraps its per-page diagrams in a NewpagedDiagram container,
            // which isn't itself a TitledDiagram. Unwrap to the first page's real
            // diagram so its actual DiagramType is reported instead of the wrapper's
            // class name.
            if (diagram instanceof NewpagedDiagram) {
                // PlantUML 1.2026+: getDiagrams() returns List<UgDiagram> (still IS-A
                // Diagram) rather than List<Diagram> as in 1.2025 and earlier.
                List<net.sourceforge.plantuml.UgDiagram> pages = ((NewpagedDiagram) diagram).getDiagrams();
                if (!pages.isEmpty()) {
                    diagram = pages.get(0);
                }
            }

            String warning = diagram.getWarningOrError();
            boolean valid = warning == null || warning.trim().isEmpty();

            String type;
            if (diagram instanceof TitledDiagram) {
                // PlantUML 1.2026+: skin.UmlDiagramType (and getUmlDiagramType()) were
                // removed; the same fine-grained values (SEQUENCE/CLASS/STATE/...) now
                // live directly on core.DiagramType, returned by getDiagramType(). This
                // still matches the DIAGRAM type names extract_insights_jar's
                // _JAR_TYPE_MAP expects (confirmed against 1.2026.6).
                type = ((TitledDiagram) diagram).getDiagramType().name();
            } else {
                type = diagram.getClass().getSimpleName();
            }

            return "<result><valid>" + valid + "</valid><type>" + type
                    + "</type><blocks>" + blocks.size() + "</blocks></result>";
        } catch (Throwable t) {
            return "<result><valid>false</valid><type>ERROR</type><blocks>0</blocks></result>";
        }
    }

    private static String readAll(InputStream in) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        in.transferTo(buffer);
        return buffer.toString("UTF-8");
    }

    /** Reads one newline-terminated line as raw bytes; returns null at EOF with no data read. */
    private static String readLine(InputStream in) throws IOException {
        ByteArrayOutputStream line = new ByteArrayOutputStream();
        int b;
        boolean any = false;
        while ((b = in.read()) != -1) {
            any = true;
            if (b == '\n') {
                return line.toString("UTF-8");
            }
            line.write(b);
        }
        return any ? line.toString("UTF-8") : null;
    }
}
