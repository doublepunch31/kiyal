import hashlib
from array import array
from collections.abc import Sequence


class NoFreeBlocksError(RuntimeError):
    """Raised when a block table needs more blocks than the pool has free."""


def block_hash(parent: bytes | None, token_ids: Sequence[int]) -> bytes:
    """Returns the content hash of a full block, which stands for every token up to its end.

    The hash covers the block's own tokens and, through the hash of the block before it, every
    token before them. Two blocks get the same hash only if they hold the same tokens after the
    same prefix, so at the same positions, and so their keys and values are the same.

    SHA-256 rather than Python's `hash`: a collision would give one prompt the keys and values
    of another, and the built-in hash of a tuple of ints can be made to collide on purpose by
    a client that crafts its prompt.

    Args:
        parent: Hash of the block before this one, None for a sequence's first block.
        token_ids: Tokens of the block.
    """
    digest = hashlib.sha256(parent or b"")
    digest.update(array("q", token_ids).tobytes())
    return digest.digest()


class BlockManager:
    """Hands out the blocks of a paged KV pool and takes them back.

    The keys and values of a sequence live in blocks of `block_size` tokens that need not be
    next to each other in the pool. The sequence's block table lists its blocks in order, so
    its token i sits at offset i % block_size of block table[i // block_size]. A sequence
    takes a block only when it needs one, instead of reserving room for the longest sequence
    it could become, so at most one partly filled block per sequence is unused (Kwon et al.,
    "PagedAttention", SOSP 2023).

    Every block has a reference count, the number of block tables that hold it, and is free
    again when the count drops to 0.

    Prefix caching. Once a full block's keys and values are computed, the caller can cache it
    under its `block_hash`. A later sequence whose first blocks hold the same tokens finds
    them with `cached_prefix` and holds the same blocks, with their counts raised, instead of
    computing their keys and values again. Only full blocks are shared. A sequence writes only
    past its last full block, so it never writes into a block that another table holds, and a
    shared block needs no copy-on-write. A block leaves the cache when no table holds it.

    `watermark` blocks are held back from new sequences: an admission must leave them free,
    so the running sequences can still grow at the next steps instead of being preempted
    right after the admission. Growing a running sequence may use them.
    """

    def __init__(self, num_blocks: int, block_size: int, watermark: int = 0) -> None:
        """Starts with every block free.

        Raises:
            ValueError: If `num_blocks` or `block_size` is below 1, or `watermark` is not in
                [0, num_blocks).
        """
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be 1 or greater, got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be 1 or greater, got {block_size}")
        if not 0 <= watermark < num_blocks:
            raise ValueError(f"watermark must be in [0, {num_blocks}), got {watermark}")
        self._block_size = block_size
        self._watermark = watermark
        # A stack whose top is block 0, so blocks are handed out in order.
        self._free = list(reversed(range(num_blocks)))
        self._ref_counts = [0] * num_blocks
        # The hash each cached block is known by, None for blocks outside the cache.
        self._hashes: list[bytes | None] = [None] * num_blocks
        self._cached: dict[bytes, int] = {}

    @property
    def num_blocks(self) -> int:
        return len(self._ref_counts)

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def watermark(self) -> int:
        return self._watermark

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def ref_count(self, block: int) -> int:
        """Number of block tables that hold `block`."""
        return self._ref_counts[block]

    def blocks_needed(self, num_tokens: int) -> int:
        """Number of blocks that `num_tokens` tokens fill, the last one maybe in part."""
        return -(-num_tokens // self._block_size)

    def cached_prefix(self, hashes: Sequence[bytes]) -> list[int]:
        """Returns the cached blocks of the longest run of `hashes` from the first one.

        Args:
            hashes: `block_hash` of each full block of a sequence, in order.
        """
        blocks: list[int] = []
        for content_hash in hashes:
            block = self._cached.get(content_hash)
            if block is None:
                break
            blocks.append(block)
        return blocks

    def cache_block(self, block: int, content_hash: bytes) -> None:
        """Lets later sequences share a full block whose keys and values are computed.

        If another block is cached under the same hash, for example because two sequences
        with the same prefix computed it in the same step, the cache keeps that one, and this
        block stays its table's own.

        Raises:
            ValueError: If no table holds the block, or it is cached under another hash.
        """
        if self._ref_counts[block] == 0:
            raise ValueError(f"block {block} is free and cannot be cached")
        current = self._hashes[block]
        if current is not None:
            if current != content_hash:
                raise ValueError(f"block {block} is cached already, under another hash")
            return
        if content_hash in self._cached:
            return
        self._cached[content_hash] = block
        self._hashes[block] = content_hash

    def can_allocate(self, num_tokens: int, cached_blocks: Sequence[int] = ()) -> bool:
        """Whether a new sequence of `num_tokens` tokens fits and leaves the watermark free.

        Args:
            num_tokens: Tokens of the sequence.
            cached_blocks: Cached blocks it shares for its first tokens, from `cached_prefix`.
        """
        new_blocks = self.blocks_needed(num_tokens) - len(cached_blocks)
        return self.num_free_blocks - new_blocks >= self._watermark

    def allocate(self, num_tokens: int, cached_blocks: Sequence[int] = ()) -> list[int]:
        """Returns a new block table with room for `num_tokens` tokens.

        The table starts with `cached_blocks`, which it shares with the tables that hold them
        already, and continues with free blocks. The watermark is not checked here;
        `can_allocate` decides whether a new sequence may start.

        Args:
            num_tokens: Tokens of the sequence.
            cached_blocks: Cached blocks it shares for its first tokens, from `cached_prefix`.

        Raises:
            ValueError: If `num_tokens` is below 1, a shared block is not cached, or there are
                more shared blocks than the tokens fill.
            NoFreeBlocksError: If fewer blocks are free than the tokens need. Nothing is
                allocated then.
        """
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be 1 or greater, got {num_tokens}")
        if len(cached_blocks) * self._block_size > num_tokens:
            raise ValueError(
                f"{len(cached_blocks)} shared blocks hold more than {num_tokens} tokens"
            )
        for block in cached_blocks:
            if self._hashes[block] is None:
                raise ValueError(f"block {block} is not cached and cannot be shared")
        missing = self.blocks_needed(num_tokens) - len(cached_blocks)
        if missing > self.num_free_blocks:
            raise NoFreeBlocksError(
                f"{num_tokens} tokens need {missing} more blocks, but {self.num_free_blocks} "
                f"are free"
            )
        for block in cached_blocks:
            self._ref_counts[block] += 1
        block_table = list(cached_blocks)
        self.append_slot(block_table, num_tokens)
        return block_table

    def can_append(self, block_table: list[int], num_tokens: int) -> bool:
        """Whether the table can grow to hold `num_tokens` tokens with the free blocks."""
        return self.blocks_needed(num_tokens) - len(block_table) <= self.num_free_blocks

    def append_slot(self, block_table: list[int], num_tokens: int) -> None:
        """Adds free blocks to the table until it has room for `num_tokens` tokens.

        A decode step adds one token, so it takes a new block only when the table's blocks
        are full.

        Raises:
            NoFreeBlocksError: If fewer blocks are free than the table needs; the table is
                then unchanged.
        """
        missing = self.blocks_needed(num_tokens) - len(block_table)
        if missing > self.num_free_blocks:
            raise NoFreeBlocksError(
                f"{num_tokens} tokens need {missing} more blocks, but {self.num_free_blocks} "
                f"are free"
            )
        for _ in range(missing):
            block = self._free.pop()
            self._ref_counts[block] = 1
            block_table.append(block)

    def free(self, block_table: list[int]) -> None:
        """Gives up the table's hold on its blocks and empties the table.

        A block becomes free once no table holds it, and leaves the cache then.

        Raises:
            ValueError: If the table holds a block that is free already; nothing is freed
                then.
        """
        for block in block_table:
            if self._ref_counts[block] == 0:
                raise ValueError(f"block {block} is free already")
        # In reverse, so the next table takes the blocks in the same order and the lowest
        # ids stay on top of the stack.
        for block in reversed(block_table):
            self._ref_counts[block] -= 1
            if self._ref_counts[block] == 0:
                content_hash = self._hashes[block]
                if content_hash is not None:
                    del self._cached[content_hash]
                    self._hashes[block] = None
                self._free.append(block)
        block_table.clear()
