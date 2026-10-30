# VIB1 frame interface

TCP port: 9000. Header format: `!4sBBH I Q B B H I I`, 32 bytes.

| Field | Bytes | Meaning |
|---|---:|---|
| magic | 4 | ASCII VIB1 |
| version / flags | 1 + 1 | version 1; flags 0 data or 1 site selection |
| header length | 2 | 32 |
| sequence | 4 | unsigned packet counter |
| timestamp | 8 | sender timestamp, nanoseconds |
| site id length | 1 | 1–63 UTF-8 bytes |
| channels | 1 | 3 for data packets |
| samples per channel | 2 | 1024 for data packets |
| sample rate | 4 | 12000 Hz |
| payload bytes | 4 | 6144 for 3×1024×int16 |

The body is the UTF-8 site id followed by little-endian int16 samples in C order.
The final four network-order bytes contain CRC32 of header and body.
For flags=1, channel/sample/rate/payload fields are zero and the body has the site id only.
The global sender counter interleaves sites; consecutive frames of a single site
therefore need not have consecutive global sequence numbers.
