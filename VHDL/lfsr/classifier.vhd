-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.
LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY classifier IS
    GENERIC (
        d       : INTEGER := 1000;  -- Dimension size + zero-padding
        c       : INTEGER := 10;    -- Number of classes
        n       : INTEGER := 7;     -- Bit-widths of memory pointer, counter, etc.
        adI     : INTEGER := 5;     -- Number of confComp modules or adder inputs (ceiling(D / 2^n))
        adz     : INTEGER := 3;     -- Zero-padding for RSA = 2**? - adI
        zComp   : INTEGER := 6;     -- Zero-padding for Mux Comp = 2**? - c
        lgCn    : INTEGER := 4;     -- Ceiling log2(number of classes)
        logn    : INTEGER := 3      -- MuxCell RSA, ceiling log2(popCounters)
    );
    PORT (
        clk         : IN  STD_LOGIC;
        rst         : IN  STD_LOGIC;
        run         : IN  STD_LOGIC;
        hv          : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
        done        : OUT STD_LOGIC;
        TLAST_S     : OUT STD_LOGIC;
        TVALID_S    : OUT STD_LOGIC;
        pointer     : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0);
        classIndex  : OUT STD_LOGIC_VECTOR(lgCn-1 DOWNTO 0)
    );
END ENTITY classifier;

ARCHITECTURE behavioral OF classifier IS

    -- Counting Simulation Top-Level Component
    COMPONENT countingSimTop IS
        GENERIC (
            n           : INTEGER := 10;  -- Bit-widths of memory pointer, counter, etc.
            d           : INTEGER := 10;  -- Number of confComp modules
            z           : INTEGER := 0;   -- Zero-padding to 2** for RSA
            classNumber : INTEGER := 10;  -- Number of classes for memory image
            logInNum    : INTEGER := 3    -- MuxCell, ceiling log2(popCounters)
        );
        PORT (
            clk      : IN  STD_LOGIC;
            rst      : IN  STD_LOGIC;
            run      : IN  STD_LOGIC;
            hv       : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
            done     : OUT STD_LOGIC;
            pointer  : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0);
            dout     : OUT STD_LOGIC_VECTOR(classNumber * (n + logInNum) - 1 DOWNTO 0)
        );
    END COMPONENT;

    -- Comparator Top-Level Component
    COMPONENT comparatorTop IS
        GENERIC (
            len  : INTEGER := 8;  -- Bit width out adder
            n    : INTEGER := 10; -- Number of classes
            z    : INTEGER := 10; -- Zero-padding to 2**
            lgn  : INTEGER := 4   -- Log2 of the number of classes
        );
        PORT (
            clk         : IN  STD_LOGIC;
            rst         : IN  STD_LOGIC;
            run         : IN  STD_LOGIC;
            a           : IN  STD_LOGIC_VECTOR(n * len - 1 DOWNTO 0);
            done        : OUT STD_LOGIC;
            TLAST_S     : OUT STD_LOGIC;
            TVALID_S    : OUT STD_LOGIC;
            classIndex  : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0)
        );
    END COMPONENT;

    -- Internal Signals
    SIGNAL hvTOcount : STD_LOGIC_VECTOR(adI - 1 DOWNTO 0);
    SIGNAL dones     : STD_LOGIC;
    SIGNAL toComp    : STD_LOGIC_VECTOR(c * (n + logn) - 1 DOWNTO 0);
    SIGNAL point     : STD_LOGIC_VECTOR(n-1 DOWNTO 0);

BEGIN

    -- Generate hvTOcount using pointer indexing
    concat: FOR I IN adI-1 DOWNTO 0 GENERATE
        hvTOcount(I) <= hv(to_integer(unsigned(point)) + (2**n) * I);
    END GENERATE concat;

    -- Counting Simulation Instance
    CST : countingSimTop
        GENERIC MAP (
            n           => n,
            d           => adI,
            z           => adz,
            classNumber => c,
            logInNum    => logn
        )
        PORT MAP (
            clk     => clk,
            rst     => rst,
            run     => run,
            hv      => hvTOcount,
            done    => dones,
            pointer => point,
            dout    => toComp
        );

    -- Comparator Instance
    CT : comparatorTop
        GENERIC MAP (
            len  => (n + logn),
            n    => c,
            z    => zComp,
            lgn  => lgCn
        )
        PORT MAP (
            clk        => clk,
            rst        => rst,
            run        => dones,
            a          => toComp,
            done       => done,
            TLAST_S    => TLAST_S,
            TVALID_S   => TVALID_S,
            classIndex => classIndex
        );

    -- Assign output pointer
    pointer <= point;

END ARCHITECTURE behavioral;

