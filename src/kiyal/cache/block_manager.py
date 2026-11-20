class NoFreeBlocksError(RuntimeError):
    """Raised when a block table needs more blocks than the pool has free."""


class BlockManager:
    """Hands out the blocks of a paged KV pool and takes them back.

    The keys and values of a sequence live in blocks of `block_size` tokens that need not be
    next to each other in the pool. The sequence's block table lists its blocks in order, so
    its token i sits at offset i % block_size of block table[i // block_size]. A sequence
    takes a block only when it needs one, instead of reserving room for the longest sequence
    it could become, so at most one partly filled block per sequence is unused (Kwon et al.,
    "PagedAttention", SOSP 2023).

    Every block has a reference count, the number of block tables that hold it, and is free
    again when the count drops to 0. Until blocks are shared between sequences, each block in
    use has exactly one holder.

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

    def can_allocate(self, num_tokens: int) -> bool:
        """Whether a new sequence of `num_tokens` tokens fits and leaves the watermark free."""
        return self.num_free_blocks - self.blocks_needed(num_tokens) >= self._watermark

    def allocate(self, num_tokens: int) -> list[int]:
        """Returns a new block table with room for `num_tokens` tokens.

        The watermark is not checked here; `can_allocate` decides whether a new sequence
        may start.

        Raises:
            ValueError: If `num_tokens` is below 1.
            NoFreeBlocksError: If fewer blocks are free than the tokens need.
        """
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be 1 or greater, got {num_tokens}")
        block_table: list[int] = []
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

        A block becomes free once no table holds it.

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
                self._free.append(block)
        block_table.clear()
